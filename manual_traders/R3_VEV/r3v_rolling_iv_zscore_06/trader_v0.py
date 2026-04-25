"""
Round 3: single-strike VEV_5000 — rolling implied-vol z-score mean reversion.

TTE / tape mapping (round3work/round3description.txt): vouchers have 7-day
deadline from round 1; each competition round is one day. Historical tape
`prices_round_3_day_{d}.csv` aligns with TTE = (8 - d) days at the start of
that tape (pattern from the spec example: historical day 1 -> 8d, day 2 -> 7d,
day 3 -> 6d). The backtester instantiates a fresh Trader per tape day with
empty traderData, so rolling IV stats reset each day.

IV: Black–Scholes European call, r=0, sigma from mid option and mid underlying
(bisection). Fixed T = 7/365 years in the solver as a single reference maturity
across tapes (rolling z uses the same convention each day).

Signals: maintain deque of IV; z = (IV - mean) / std (population std, floor).
Short voucher when z > ENTRY (rich vol); long when z < -ENTRY. Flatten when
|z| < EXIT. Trade sizes clip to position limits; take at touch (bid/ask).
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
# Reference TTE for BS (see module docstring)
T_YEAR = 7.0 / 365.0

ROLL_WIN = 400
Z_ENTRY = 2.0
Z_EXIT = 0.35
ORDER_Q = 18
MIN_STD = 5e-4


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
    bid = max(depth.buy_orders)
    ask = min(depth.sell_orders)
    return bid, ask


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}

        iv_hist: deque[float] = deque(
            (float(x) for x in td.get("_iv", [])), maxlen=ROLL_WIN + 50
        )
        result: dict[str, list[Order]] = {p: [] for p in LIMITS}

        pos = state.position
        und = state.order_depths.get(UNDER)
        vev = state.order_depths.get(VEV_TARGET)
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
        iv = implied_vol(s_mid, STRIKE, T_YEAR, c_mid)
        if iv is not None:
            iv_hist.append(iv)

        spread_vev = va - vb
        spread_u = ua - ub

        z = 0.0
        if len(iv_hist) >= 30:
            mu = sum(iv_hist) / len(iv_hist)
            var = sum((x - mu) ** 2 for x in iv_hist) / max(len(iv_hist), 1)
            sig = math.sqrt(var)
            if sig < MIN_STD:
                sig = MIN_STD
            if iv is not None:
                z = (iv - mu) / sig

        q = pos.get(VEV_TARGET, 0)
        lim = LIMITS[VEV_TARGET]

        orders_v: list[Order] = []
        if iv is not None and len(iv_hist) >= 30:
            if q > 0 and z < Z_EXIT:
                # long voucher, vol normalized — sell at bid (lift bid side volume)
                sell_px = vb
                avail = vev.buy_orders.get(sell_px, 0) if sell_px is not None else 0
                dq = min(q, ORDER_Q, avail)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, sell_px, -dq))
            elif q < 0 and z > -Z_EXIT:
                buy_px = va
                avail = abs(vev.sell_orders.get(buy_px, 0)) if buy_px is not None else 0
                dq = min(-q, ORDER_Q, avail)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, buy_px, dq))
            elif z > Z_ENTRY and q <= 0:
                # rich IV — fade (short voucher at bid)
                sell_px = vb
                avail = vev.buy_orders.get(sell_px, 0) if sell_px is not None else 0
                room = lim + q
                dq = min(ORDER_Q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, sell_px, -dq))
            elif z < -Z_ENTRY and q >= 0:
                buy_px = va
                avail = abs(vev.sell_orders.get(buy_px, 0)) if buy_px is not None else 0
                room = lim - q
                dq = min(ORDER_Q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, buy_px, dq))

        result[VEV_TARGET] = orders_v

        # Light extract liquidity provision when flat on voucher (microstructure)
        uq = pos.get(UNDER, 0)
        if abs(q) < 5 and spread_u <= 8 and abs(uq) < 80:
            if uq > 0 and ub is not None:
                avail = ub in und.buy_orders and und.buy_orders[ub] or 0
                dq = min(uq, 12, avail)
                if dq > 0:
                    result[UNDER].append(Order(UNDER, ub, -dq))
            elif uq < 0 and ua is not None:
                avail = abs(und.sell_orders.get(ua, 0))
                dq = min(-uq, 12, avail)
                if dq > 0:
                    result[UNDER].append(Order(UNDER, ua, dq))

        td["_iv"] = list(iv_hist)[-ROLL_WIN - 20 :]
        td["_z"] = z
        td["_spr_v"] = spread_vev
        td["_spr_u"] = spread_u
        return result, 0, json.dumps(td)
