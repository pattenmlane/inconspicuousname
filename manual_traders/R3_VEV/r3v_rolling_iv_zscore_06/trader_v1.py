"""
v1 vs v0:
- Rolling z uses **past IV only** (mean/std over deque before appending current),
  avoiding self-inflation of the denominator when the current print is extreme.
- Black–Scholes **delta** on VEV_5000 used to hedge extract when voucher
  position is non-zero (single-strike thesis, grounded in greeks).
- Slightly tighter entry (1.75), smaller clip, wider rolling window.
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

ROLL_WIN = 500
Z_ENTRY = 1.75
Z_EXIT = 0.45
ORDER_Q = 14
MIN_STD = 6e-4
HEDGE_FRAC = 0.85  # scale delta hedge to avoid limit churn


def _N_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_delta(spot: float, strike: float, t: float, vol: float) -> float | None:
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return None
    sig_rt = vol * math.sqrt(t)
    if sig_rt < 1e-12:
        return None
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / sig_rt
    return _N_cdf(d1)


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
        result: dict[str, list[Order]] = {p: [] for p in LIMITS}

        und = state.order_depths.get(UNDER)
        vev = state.order_depths.get(VEV_TARGET)
        pos = state.position
        if und is None or vev is None:
            td["_iv"] = list(iv_hist)
            return result, 0, json.dumps(td)

        ub, ua = _touch(und)
        vb, va = _touch(vev)
        if ub is None or ua is None or vb is None or va is None:
            td["_iv"] = list(iv_hist)
            return result, 0, json.dumps(td)

        s_mid = 0.5 * (ub + ua)
        c_mid = 0.5 * (vb + va)
        iv_now = implied_vol(s_mid, STRIKE, T_YEAR, c_mid)

        z = 0.0
        if iv_now is not None and len(iv_hist) >= 40:
            mu = sum(iv_hist) / len(iv_hist)
            var = sum((x - mu) ** 2 for x in iv_hist) / max(len(iv_hist), 1)
            sig = math.sqrt(var)
            if sig < MIN_STD:
                sig = MIN_STD
            z = (iv_now - mu) / sig

        if iv_now is not None:
            iv_hist.append(iv_now)

        q = pos.get(VEV_TARGET, 0)
        lim = LIMITS[VEV_TARGET]
        orders_v: list[Order] = []

        if iv_now is not None and len(iv_hist) >= 41:
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
            elif z > Z_ENTRY and q <= 0:
                avail = vev.buy_orders.get(vb, 0)
                room = lim + q
                dq = min(ORDER_Q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, vb, -dq))
            elif z < -Z_ENTRY and q >= 0:
                avail = abs(vev.sell_orders.get(va, 0))
                room = lim - q
                dq = min(ORDER_Q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, va, dq))

        result[VEV_TARGET] = orders_v

        # Delta hedge extract vs voucher (long call delta; short voucher -> long delta units underlying)
        u_pos = pos.get(UNDER, 0)
        if iv_now is not None and abs(q) >= 3:
            dlt = bs_call_delta(s_mid, STRIKE, T_YEAR, iv_now)
            if dlt is not None:
                hedge_units = int(round(HEDGE_FRAC * abs(q) * dlt))
                hedge_units = max(1, min(hedge_units, 40))
                if q < 0:
                    # short calls -> want long underlying
                    if u_pos < LIMITS[UNDER] - 5:
                        buy_px = ua
                        avail = abs(und.sell_orders.get(buy_px, 0))
                        hq = min(hedge_units, avail, LIMITS[UNDER] - u_pos, 25)
                        if hq > 0:
                            result[UNDER].append(Order(UNDER, buy_px, hq))
                elif q > 0:
                    if u_pos > -LIMITS[UNDER] + 5:
                        sell_px = ub
                        avail = und.buy_orders.get(sell_px, 0)
                        hq = min(hedge_units, avail, u_pos + LIMITS[UNDER], 25)
                        if hq > 0:
                            result[UNDER].append(Order(UNDER, sell_px, -hq))

        td["_iv"] = list(iv_hist)[-ROLL_WIN - 50 :]
        td["_z"] = z
        return result, 0, json.dumps(td)
