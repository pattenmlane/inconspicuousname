"""
Round 3: wide passive book on deep ITM vouchers (VEV_4000, VEV_4500) with tiny size.

Tape analysis (see analysis.json): top-of-book spreads ~21 / ~16 vs ~2 tick penny gap —
posting multiple ticks inside the spread still earns edge if filled. BS delta from
implied mids is high (~0.9+), so extract is used as a light delta hedge via quote skew.
TTE for BS: historical Round-3 files day 0,1,2 align to TTE 7,6,5 calendar days per
round3work/round3description.txt; backtest reloads Trader each day — we snapshot TTE
from the first tick's timestamp==0 convention (day-local) is unavailable, so we use
TTE=6d as a mid-maturity proxy for greek scale (documented in results).
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Dict, List, Optional

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
STRIKE = {"VEV_4000": 4000, "VEV_4500": 4500}
TARGET_VEV = ("VEV_4000", "VEV_4500")
# Mid-maturity proxy when day index is not in TradingState (see module docstring).
TTE_DAYS_PROXY = 6
DAYS_PER_YEAR = 365

# Passive: only when spread is huge (tape ~16–21); grid baseline.
WIDE_SPREAD_MIN = 18
DEEP_OFFSET = {"VEV_4000": 7, "VEV_4500": 5}
PASSIVE_SIZE = 2

# Extract: slow fair + inventory skew
EXTRACT_EMA_ALPHA = 0.02
EXTRACT_PASSIVE_SIZE = 6
HYDRO_EMA_ALPHA = 0.02
HYDRO_PASSIVE_SIZE = 2
HYDRO_WIDE_MIN = 16


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


def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return _N.cdf(d1)


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

        od_u = state.order_depths.get(UNDER)
        mid_u = _mid(od_u) if od_u else None
        e_ema = data.get("e_ema")
        if mid_u is not None:
            if e_ema is None:
                e_ema = mid_u
            else:
                e_ema = (1.0 - EXTRACT_EMA_ALPHA) * float(e_ema) + EXTRACT_EMA_ALPHA * mid_u
        data["e_ema"] = e_ema

        od_h = state.order_depths.get(HYDRO)
        mid_h = _mid(od_h) if od_h else None
        h_ema = data.get("h_ema")
        if mid_h is not None:
            if h_ema is None:
                h_ema = mid_h
            else:
                h_ema = (1.0 - HYDRO_EMA_ALPHA) * float(h_ema) + HYDRO_EMA_ALPHA * mid_h
        data["h_ema"] = h_ema

        T = TTE_DAYS_PROXY / DAYS_PER_YEAR
        vev_iv: Dict[str, Optional[float]] = {}
        vev_delta: Dict[str, float] = {}
        if mid_u is not None:
            for sym in TARGET_VEV:
                od = state.order_depths.get(sym)
                if od is None:
                    continue
                m = _mid(od)
                if m is None:
                    continue
                K = STRIKE[sym]
                iv = bs_call_iv(mid_u, K, T, m)
                vev_iv[sym] = iv
                if iv is not None:
                    vev_delta[sym] = bs_call_delta(mid_u, K, T, iv)
                else:
                    vev_delta[sym] = 0.0

        net_call_delta = sum(_pos(state, s) * vev_delta.get(s, 0.0) for s in TARGET_VEV)
        skew = int(round(net_call_delta * 0.35))
        skew = max(-8, min(8, skew))

        # --- VEV_4000 / VEV_4500: deep passive inside wide spread ---
        for sym in TARGET_VEV:
            od = state.order_depths.get(sym)
            if od is None:
                continue
            sp = _spread(od)
            if sp is None or sp < WIDE_SPREAD_MIN:
                continue
            bb, ba = _best_bid_ask(od)
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
                q = min(PASSIVE_SIZE, buy_room)
                orders[sym].append(Order(sym, int(bid_px), q))
            if sell_room > 0:
                q = min(PASSIVE_SIZE, sell_room)
                orders[sym].append(Order(sym, int(ask_px), -q))

        # --- Extract: penny improve around EMA, inventory + call-delta skew ---
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
                    q = min(EXTRACT_PASSIVE_SIZE, buy_room)
                    orders[UNDER].append(Order(UNDER, bid_px, q))
                if sell_room > 0 and ask_px > bb:
                    q = min(EXTRACT_PASSIVE_SIZE, sell_room)
                    orders[UNDER].append(Order(UNDER, ask_px, -q))

        # --- Hydrogel: tiny MM when spread wide (tape ~16) ---
        if od_h is not None and h_ema is not None:
            sp = _spread(od_h)
            if sp is not None and sp >= HYDRO_WIDE_MIN:
                bb, ba = _best_bid_ask(od_h)
                if bb is not None and ba is not None:
                    pos = _pos(state, HYDRO)
                    lim = LIMITS[HYDRO]
                    fair = float(h_ema)
                    bid_px = min(bb + 1, int(math.floor(fair)) - 1)
                    ask_px = max(ba - 1, int(math.ceil(fair)) + 1)
                    buy_room = lim - pos
                    sell_room = lim + pos
                    if buy_room > 0 and bid_px > 0 and bid_px < ba:
                        q = min(HYDRO_PASSIVE_SIZE, buy_room)
                        orders[HYDRO].append(Order(HYDRO, bid_px, q))
                    if sell_room > 0 and ask_px > bb:
                        q = min(HYDRO_PASSIVE_SIZE, sell_room)
                        orders[HYDRO].append(Order(HYDRO, ask_px, -q))

        return orders, 0, json.dumps(data)
