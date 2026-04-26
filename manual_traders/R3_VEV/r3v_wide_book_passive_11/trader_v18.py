"""trader_v18: vouchers_final_strategy (Round 3) — v17 + joint-gate *long lean* on extract.

Parent: trader_v17 (joint VEV_5200+VEV_5300 L1 both <=2 = risk_on; wide passive 4000/4500;
penny extract MM; 4000/4500 delta+vega skew; no HYDRO; 5200/5300 signal-only).

**Change vs v17 (STRATEGY.txt optional read):** when `risk_on`, tape analysis shows higher
mean K-step *forward* extract *mid* change — a toy policy is to lean long extract for short
holds. We do **not** hit mid; we only **asymmetric** passive sizes: +1 on extract **buys** and
-1 on extract **sells** when the gate is on (capped 12 / floored 1), keeping the same quote
prices as v17. This is execution-layer; mid edge can still vanish under worse-matching (caveat
in STRATEGY.txt).

**inclineGod / Sonic:** we log `s_5200`, `s_5300` each tick (spread = ask1 - bid1) for
post-hoc "corr of spreads" work; `risk_on_5200_5300` is the Sonic joint gate. Same `gate_ret_ema`
as v17.
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

_N = NormalDist()

UNDER = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
LIMITS: Dict[str, int] = {
    HYDRO: 200,
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
GATE_5200 = "VEV_5200"
GATE_5300 = "VEV_5300"
STRIKE = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
}
TARGET_VEV = ("VEV_4000", "VEV_4500")
SKEW_VEV = ("VEV_4000", "VEV_4500")

TTE_DAYS_PROXY = 6
DAYS_PER_YEAR = 365

WIDE_SPREAD_MIN = 15
DEEP_OFFSET = {"VEV_4000": 4, "VEV_4500": 5}
PASSIVE_SIZE = 3

TIGHT_MAX_SPREAD = 2
GATE_EXTRACT_MULT = 1.0

EXTRACT_EMA_ALPHA = 0.02
EXTRACT_PASSIVE_SIZE = 7
# When risk_on: want larger buy size, smaller offer size (long lean) vs symmetric ex_sz
RISK_ON_BUY_BUMP = 1
RISK_ON_SELL_TRIM = 1

GATE_RET_EMA_ALPHA = 0.12

DELTA_SKEW = 0.3
VEGA_SKEW = 0.0005
SKEW_CAP = 8


def _pos(state: TradingState, sym: str) -> int:
    return int(state.position.get(sym, 0))


def _mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return (bb + ba) / 2.0


def _best_bid_ask(od: OrderDepth) -> tuple[Optional[int], Optional[int]]:
    if not od.buy_orders or not od.sell_orders:
        return None, None
    return max(od.buy_orders.keys()), min(od.sell_orders.keys())


def _spread(od: OrderDepth) -> Optional[int]:
    bb, ba = _best_bid_ask(od)
    if bb is None or ba is None:
        return None
    return ba - bb


def _joint_gate(
    state: TradingState,
) -> Tuple[bool, Optional[int], Optional[int]]:
    o52 = state.order_depths.get(GATE_5200)
    o53 = state.order_depths.get(GATE_5300)
    if o52 is None or o53 is None:
        return False, None, None
    s52 = _spread(o52)
    s53 = _spread(o53)
    if s52 is None or s53 is None:
        return False, s52, s53
    tight = s52 <= TIGHT_MAX_SPREAD and s53 <= TIGHT_MAX_SPREAD
    return tight, s52, s53


def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return _N.cdf(d1)


def bs_call_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return S * _N.pdf(d1) * math.sqrt(T)


def bs_call_iv(S: float, K: float, T: float, mid: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-6:
        return None
    lo, hi = 1e-6, 5.0
    for _ in range(55):
        m = 0.5 * (lo + hi)

        def price(sig: float) -> float:
            if sig <= 0:
                return intrinsic
            vv = sig * math.sqrt(T)
            d1 = (math.log(S / K) + 0.5 * vv * vv) / vv
            d2 = d1 - vv
            return S * _N.cdf(d1) - K * _N.cdf(d2)

        if price(m) > mid:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


class Trader:
    def run(self, state: TradingState):
        data: dict = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except json.JSONDecodeError:
                data = {}

        orders: Dict[str, List[Order]] = {p: [] for p in LIMITS}

        risk_on, s52, s53 = _joint_gate(state)
        data["risk_on_5200_5300"] = risk_on
        data["s_5200"] = s52
        data["s_5300"] = s53

        od_u = state.order_depths.get(UNDER)
        mid_u = _mid(od_u) if od_u else None
        ext_prev = data.get("ext_mid_prev")
        gate_ret_ema = data.get("gate_ret_ema")
        if gate_ret_ema is None:
            gate_ret_ema = 0.0
        if (
            risk_on
            and mid_u is not None
            and ext_prev is not None
            and isinstance(ext_prev, (int, float))
        ):
            r_tick = float(mid_u) - float(ext_prev)
            gate_ret_ema = (1.0 - GATE_RET_EMA_ALPHA) * float(
                gate_ret_ema
            ) + GATE_RET_EMA_ALPHA * r_tick
        data["gate_ret_ema"] = gate_ret_ema
        if mid_u is not None:
            data["ext_mid_prev"] = float(mid_u)

        e_ema = data.get("e_ema")
        if mid_u is not None:
            if e_ema is None:
                e_ema = mid_u
            else:
                e_ema = (1.0 - EXTRACT_EMA_ALPHA) * float(
                    e_ema
                ) + EXTRACT_EMA_ALPHA * mid_u
        data["e_ema"] = e_ema

        T = TTE_DAYS_PROXY / DAYS_PER_YEAR
        vev_delta: Dict[str, float] = {}
        vev_vega: Dict[str, float] = {}
        if mid_u is not None:
            for sym in SKEW_VEV:
                odv = state.order_depths.get(sym)
                if odv is None:
                    continue
                m = _mid(odv)
                if m is None:
                    continue
                K = STRIKE[sym]
                iv = bs_call_iv(mid_u, K, T, m)
                if iv is not None:
                    vev_delta[sym] = bs_call_delta(mid_u, K, T, iv)
                    vev_vega[sym] = bs_call_vega(mid_u, K, T, iv)
                else:
                    vev_delta[sym] = 0.0
                    vev_vega[sym] = 0.0

        net_call_delta = sum(
            _pos(state, s) * vev_delta.get(s, 0.0) for s in SKEW_VEV
        )
        net_opt_vega = sum(
            _pos(state, s) * vev_vega.get(s, 0.0) for s in SKEW_VEV
        )
        skew = int(
            round(
                net_call_delta * DELTA_SKEW + net_opt_vega * VEGA_SKEW,
            )
        )
        skew = max(-SKEW_CAP, min(SKEW_CAP, skew))

        ps = PASSIVE_SIZE
        ex_base = min(
            12,
            int(
                round(
                    EXTRACT_PASSIVE_SIZE
                    * (GATE_EXTRACT_MULT if risk_on else 1.0)
                )
            ),
        )
        ex_buy = min(12, ex_base + (RISK_ON_BUY_BUMP if risk_on else 0))
        ex_sell = max(1, ex_base - (RISK_ON_SELL_TRIM if risk_on else 0))

        for sym in TARGET_VEV:
            odv = state.order_depths.get(sym)
            if odv is None:
                continue
            sp = _spread(odv)
            if sp is None or sp < WIDE_SPREAD_MIN:
                continue
            bb, ba = _best_bid_ask(odv)
            if bb is None or ba is None:
                continue
            pos = _pos(state, sym)
            lim = LIMITS[sym]
            off = DEEP_OFFSET[sym]
            bid_px = bb + off
            ask_px = ba - off
            if bid_px >= ask_px:
                continue
            buy_room = lim - pos
            sell_room = lim + pos
            if buy_room > 0:
                q = min(ps, buy_room)
                orders[sym].append(Order(sym, int(bid_px), q))
            if sell_room > 0:
                q = min(ps, sell_room)
                orders[sym].append(Order(sym, int(ask_px), -q))

        if od_u is not None and e_ema is not None:
            bb, ba = _best_bid_ask(od_u)
            if bb is not None and ba is not None:
                pos = _pos(state, UNDER)
                lim = LIMITS[UNDER]
                fair = float(e_ema)
                bid_px = min(bb + 1, int(math.floor(fair)) - 1 - skew)
                ask_px = max(ba - 1, int(math.ceil(fair)) + 1 - skew)
                buy_room = lim - pos
                sell_room = lim + pos
                if buy_room > 0 and bid_px > 0 and bid_px < ba:
                    q = min(ex_buy, buy_room)
                    orders[UNDER].append(Order(UNDER, bid_px, q))
                if sell_room > 0 and ask_px > bb:
                    q = min(ex_sell, sell_room)
                    orders[UNDER].append(Order(UNDER, ask_px, -q))

        return orders, 0, json.dumps(data)
