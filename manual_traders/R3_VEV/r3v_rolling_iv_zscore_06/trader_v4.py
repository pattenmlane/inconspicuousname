"""
v4: Tuned rolling-IV z mean reversion on VEV_5000 (single strike).

Grid sweep on v2 family (past-only IV window, tight-spread filter, symmetric
fade): best aggregate PnL on Round 3 days 0–2 in the sweep was near
Z_ENTRY = 5.5 with HYST = 0.12, MAX_SPREAD = 5, ROLL_WIN = 480 (same shell as
v2; see analysis.json for the parameter grid).

TTE note (round3description.txt + file naming): tape `day` column 0,1,2 in
ROUND_3 prices maps to TTE 8d,7d,6d at tape start; BS reference uses T=7/365
as mid-round anchor (documented in analysis.json).
"""

from __future__ import annotations

import json
import math
from collections import deque
from datamodel import Order, OrderDepth, TradingState

VEV_TARGET = "VEV_5000"
UNDER = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"

LIMITS = {
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

STRIKE = 5000
T_YEAR = 7.0 / 365.0

ROLL_WIN = 480
Z_ENTRY = 5.5
Z_EXIT = 0.4
HYST = 0.12
MAX_SPREAD = 5
ORDER_Q = 12
MIN_STD = 6e-4


def _N_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot: float, strike: float, t: float, vol: float) -> float:
    if t <= 0 or vol <= 0:
        return max(spot - strike, 0.0)
    sig_rt = vol * math.sqrt(t)
    if sig_rt < 1e-12:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / sig_rt
    d2 = d1 - sig_rt
    return spot * _N_cdf(d1) - strike * _N_cdf(d2)


def implied_vol(spot: float, strike: float, t: float, price: float) -> float | None:
    intrinsic = max(spot - strike, 0.0)
    if price <= intrinsic + 1e-6:
        return None
    lo, hi = 1e-4, 4.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        th = bs_call_price(spot, strike, t, mid)
        if th > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _touch(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}

        iv_hist: deque[float] = deque(
            (float(x) for x in td.get("_iv", [])), maxlen=ROLL_WIN + 100
        )
        prev_z = float(td.get("_prev_z", 0.0))
        result: dict[str, list[Order]] = {p: [] for p in LIMITS}

        und = state.order_depths.get(UNDER)
        vev = state.order_depths.get(VEV_TARGET)
        pos = state.position
        if und is None or vev is None:
            td["_iv"] = list(iv_hist)
            td["_prev_z"] = prev_z
            return result, 0, json.dumps(td)

        ub, ua = _touch(und)
        vb, va = _touch(vev)
        if ub is None or ua is None or vb is None or va is None:
            td["_iv"] = list(iv_hist)
            td["_prev_z"] = prev_z
            return result, 0, json.dumps(td)

        s_mid = 0.5 * (ub + ua)
        c_mid = 0.5 * (vb + va)
        iv_now = implied_vol(s_mid, STRIKE, T_YEAR, c_mid)

        z = 0.0
        if iv_now is not None and len(iv_hist) >= 50:
            mu = sum(iv_hist) / len(iv_hist)
            var = sum((x - mu) ** 2 for x in iv_hist) / max(len(iv_hist), 1)
            sig = math.sqrt(var)
            if sig < MIN_STD:
                sig = MIN_STD
            z = (iv_now - mu) / sig

        if iv_now is not None:
            iv_hist.append(iv_now)

        spread_ok = (va - vb) <= MAX_SPREAD
        q = pos.get(VEV_TARGET, 0)
        lim = LIMITS[VEV_TARGET]
        orders_v: list[Order] = []

        crossed_hi = z > Z_ENTRY and prev_z <= Z_ENTRY - HYST
        crossed_lo = z < -Z_ENTRY and prev_z >= -Z_ENTRY + HYST

        if iv_now is not None and len(iv_hist) >= 51 and spread_ok:
            if q > 0 and z < Z_EXIT:
                avail = vev.buy_orders.get(vb, 0)
                dq = min(q, ORDER_Q, avail)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, vb, -dq))
            elif q < 0 and z > -Z_EXIT:
                avail = abs(vev.sell_orders.get(va, 0))
                dq = min(-q, ORDER_Q, avail)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, va, dq))
            elif crossed_hi and q <= 0:
                avail = vev.buy_orders.get(vb, 0)
                room = lim + q
                dq = min(ORDER_Q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, vb, -dq))
            elif crossed_lo and q >= 0:
                avail = abs(vev.sell_orders.get(va, 0))
                room = lim - q
                dq = min(ORDER_Q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, va, dq))

        result[VEV_TARGET] = orders_v

        td["_iv"] = list(iv_hist)[-ROLL_WIN - 50 :]
        td["_prev_z"] = z
        return result, 0, json.dumps(td)
