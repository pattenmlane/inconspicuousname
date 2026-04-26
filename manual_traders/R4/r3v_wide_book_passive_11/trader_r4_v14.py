"""Round 4 trader v14 — v10 with **larger lift** (4 units) and **more buy room**.

vs `trader_r4_v10`: **AGGR_BUY_Q=4**, **AGGR_MIN_BUY_ROOM=10** (cooldown 25,
max pos 160). Tests whether extra size on Tier-A buy-at-ask improves worse PnL
without blowing limit checks.

See `results.json` iteration 16.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState

UNDER = "VELVETFRUIT_EXTRACT"
GATE_5200 = "VEV_5200"
GATE_5300 = "VEV_5300"
VEV_4000 = "VEV_4000"

MARK_BUY = "Mark 67"
MARK_SELL = "Mark 22"

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    UNDER: 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

SPREAD_TIGHT_TH = 2
EXTRACT_EMA_ALPHA = 0.02
EXTRACT_SIZE_TIGHT = 7
EXTRACT_SIZE_WIDE = 4

AGGR_BUY_Q = 4
AGGR_MIN_BUY_ROOM = 10
AGGR_MAX_POSITION = 160
AGGR_COOLDOWN = 25

WIDE_V4000_MIN = 15
DEEP_OFF_4000 = 4
SIZE_4000_TIGHT = 2


def _pos(state: TradingState, sym: str) -> int:
    return int(state.position.get(sym, 0))


def _best(od: OrderDepth) -> tuple[Optional[int], Optional[int]]:
    if not od.buy_orders or not od.sell_orders:
        return None, None
    return max(od.buy_orders.keys()), min(od.sell_orders.keys())


def _spread(od: OrderDepth) -> Optional[int]:
    bb, ba = _best(od)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def _joint_tight(state: TradingState) -> bool:
    o52 = state.order_depths.get(GATE_5200)
    o53 = state.order_depths.get(GATE_5300)
    if o52 is None or o53 is None:
        return False
    s52, s53 = _spread(o52), _spread(o53)
    if s52 is None or s53 is None:
        return False
    return s52 <= SPREAD_TIGHT_TH and s53 <= SPREAD_TIGHT_TH


def _sig_67_22_extract(state: TradingState) -> bool:
    for t in state.market_trades.get(UNDER, []):
        b = getattr(t, "buyer", None) or ""
        s = getattr(t, "seller", None) or ""
        if b == MARK_BUY and s == MARK_SELL:
            return True
    return False


class Trader:
    def run(self, state: TradingState):
        data: dict = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except json.JSONDecodeError:
                data = {}

        orders: Dict[str, List[Order]] = {p: [] for p in LIMITS}

        tight = _joint_tight(state)
        sig = _sig_67_22_extract(state)
        data["sonic_tight"] = tight
        data["sig_67_22_extract"] = sig

        od_u = state.order_depths.get(UNDER)
        mid_u = None
        if od_u and od_u.buy_orders and od_u.sell_orders:
            bb, ba = _best(od_u)
            if bb is not None and ba is not None:
                mid_u = (bb + ba) / 2.0

        e_ema = data.get("e_ema")
        if mid_u is not None:
            e_ema = mid_u if e_ema is None else (1.0 - EXTRACT_EMA_ALPHA) * float(e_ema) + EXTRACT_EMA_ALPHA * mid_u
        data["e_ema"] = e_ema

        ex_sz = EXTRACT_SIZE_TIGHT if tight else EXTRACT_SIZE_WIDE

        if od_u is not None and e_ema is not None:
            bb, ba = _best(od_u)
            if bb is not None and ba is not None:
                pos = _pos(state, UNDER)
                lim = LIMITS[UNDER]
                fair = float(e_ema)
                skew = 0
                bid_px = min(bb + 1, int(math.floor(fair)) - 1 - skew)
                ask_px = max(ba - 1, int(math.ceil(fair)) + 1 - skew)
                br = lim - pos
                sr = lim + pos

                aggr_q = 0
                last_aggr = int(data.get("last_aggr_ts", -10**9))
                cooldown_ok = (state.timestamp - last_aggr) >= AGGR_COOLDOWN
                if tight and sig and cooldown_ok and pos <= AGGR_MAX_POSITION and br >= AGGR_MIN_BUY_ROOM:
                    aggr_q = min(AGGR_BUY_Q, br)
                    if aggr_q > 0 and ba > bb:
                        orders[UNDER].append(Order(UNDER, int(ba), int(aggr_q)))
                        data["last_aggr_ts"] = state.timestamp
                        data["aggr_buy_last"] = int(aggr_q)

                br_passive = br - aggr_q
                if br_passive > 0 and bid_px > 0 and bid_px < ba:
                    orders[UNDER].append(Order(UNDER, bid_px, min(ex_sz, br_passive)))
                if sr > 0 and ask_px > bb:
                    orders[UNDER].append(Order(UNDER, ask_px, -min(ex_sz, sr)))

        if tight:
            od4 = state.order_depths.get(VEV_4000)
            if od4 is not None:
                sp = _spread(od4)
                if sp is not None and sp >= WIDE_V4000_MIN:
                    bb, ba = _best(od4)
                    if bb is not None and ba is not None:
                        bid_px = bb + DEEP_OFF_4000
                        ask_px = ba - DEEP_OFF_4000
                        if bid_px < ask_px:
                            p = _pos(state, VEV_4000)
                            lim = LIMITS[VEV_4000]
                            q = SIZE_4000_TIGHT
                            if lim - p > 0:
                                orders[VEV_4000].append(Order(VEV_4000, bid_px, min(q, lim - p)))
                            if lim + p > 0:
                                orders[VEV_4000].append(Order(VEV_4000, ask_px, -min(q, lim + p)))

        return orders, 0, json.dumps(data)
