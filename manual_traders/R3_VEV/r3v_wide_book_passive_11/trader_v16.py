"""v16: wide-book passive 4000/4500 + VELVETFRUIT_EXTRACT, joint 5200/5300 tight-spread risk-on gate (shared thesis).

When VEV_5200 and VEV_5300 both have L1 spread <= TIGHT_MAX_SPREAD (2 ticks) at the same
timestamp, treat as **risk_on** (per shared thesis: favorable extract short-horizon regime):
scale VEV_4000/4500 passive size and EXTRACT size. We do **not** place orders on 5200/5300
(backtest: touch/marketable quotes there were strongly negative under worse-matching).
Greeks for 5200/5300 still feed the extract delta+vega skew.

**No HYDROGEL_PACK** orders: PnL focus on vouchers + extract per current objective.

Greeks: BS IV from mids for delta+vega skew on 4000/4500 only (v8); T=6/365y proxy in-file.
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState

_N = NormalDist()

UNDER = "VELVETFRUIT_EXTRACT"
# Keep LIMITS for all products the engine may have; we simply do not place HYDRO orders.
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
# Deep passive on wide-spread 4000/4500
TARGET_VEV = ("VEV_4000", "VEV_4500")
# Skew/hedge on held positions: only 4000/4500 (v8), not 5200/5300 (gate is observational).
SKEW_VEV = ("VEV_4000", "VEV_4500")
TTE_DAYS_PROXY = 6
DAYS_PER_YEAR = 365

WIDE_SPREAD_MIN = 15
DEEP_OFFSET = {"VEV_4000": 4, "VEV_4500": 5}
PASSIVE_SIZE = 3

# Joint gate: both legs tight (inclusive)
TIGHT_MAX_SPREAD = 2
# Conservative: large mult on 4000+extract hurt in worse-matching; nudge extract only.
# Set to 1.0 to retain v8 PnL while logging gate; tune >1.0 only after re-validation in worse regime.
GATE_EXTRACT_MULT = 1.0
# Keep deep passive on 4000/4500 at v8 size when risk_on (sizing 5200+5300 is observational)

EXTRACT_EMA_ALPHA = 0.02
EXTRACT_PASSIVE_SIZE = 6

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


def _tight_risk_on(state: TradingState) -> bool:
    o52 = state.order_depths.get(GATE_5200)
    o53 = state.order_depths.get(GATE_5300)
    if o52 is None or o53 is None:
        return False
    s52 = _spread(o52)
    s53 = _spread(o53)
    if s52 is None or s53 is None:
        return False
    return s52 <= TIGHT_MAX_SPREAD and s53 <= TIGHT_MAX_SPREAD


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

        risk_on = _tight_risk_on(state)
        data["risk_on_5200_5300"] = risk_on

        od_u = state.order_depths.get(UNDER)
        mid_u = _mid(od_u) if od_u else None
        e_ema = data.get("e_ema")
        if mid_u is not None:
            if e_ema is None:
                e_ema = mid_u
            else:
                e_ema = (1.0 - EXTRACT_EMA_ALPHA) * float(e_ema) + EXTRACT_EMA_ALPHA * mid_u
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
        ex_sz = min(
            12,
            int(
                round(
                    EXTRACT_PASSIVE_SIZE * (GATE_EXTRACT_MULT if risk_on else 1.0)
                )
            ),
        )

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
                    q = min(ex_sz, buy_room)
                    orders[UNDER].append(Order(UNDER, bid_px, q))
                if sell_room > 0 and ask_px > bb:
                    q = min(ex_sz, sell_room)
                    orders[UNDER].append(Order(UNDER, ask_px, -q))

        return orders, 0, json.dumps(data)
