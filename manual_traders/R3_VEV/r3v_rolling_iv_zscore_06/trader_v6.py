"""
v6: VEV_5000 single-strike, rolling-IV z on **raw** 5000 IV (same as v5) plus a
**smile** filter from cross-strike mids (IV/Greek context, no extra orders).

Rolling z uses past-only deque of Black-Scholes IV(5000) — mean reversion signal.
Additionally require IV(5000) <= median(IV(4500), IV(5100), IV(5200), IV(5300))
when at least two neighbor IVs exist, so we only buy “cheap on the smile”
extremes, not a global level shift.

TTE: round3work/round3description — ROUND_3 day 0/1/2 ~ 8d/7d/6d at tape start; BS
uses T=7/365 as fixed reference (same as v5).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import deque
from datamodel import Order, OrderDepth, TradingState

VEV_TARGET = "VEV_5000"
UNDER = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
NEIGHBOR_SYMS = ("VEV_4500", "VEV_5100", "VEV_5200", "VEV_5300")

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
Z_ENTRY = 6.0
Z_EXIT = 0.2
MAX_SPREAD = 4
ORDER_Q = 10
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


def cheap_on_smile(spot: float, mids: dict[str, float], k0: int) -> bool | None:
    """True if IV(k0) <= neighbor median; None if not enough data to say."""
    c = mids.get(VEV_TARGET)
    if c is None:
        return None
    iv0 = implied_vol(spot, k0, T_YEAR, c)
    if iv0 is None:
        return None
    neigh: list[float] = []
    for sym in NEIGHBOR_SYMS:
        kk = int(sym.split("_")[1])
        mm = mids.get(sym)
        if mm is None:
            continue
        v = implied_vol(spot, kk, T_YEAR, mm)
        if v is not None:
            neigh.append(v)
    if len(neigh) < 2:
        return None
    return iv0 <= statistics.median(neigh)


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

        mids: dict[str, float] = {}
        for sym in (VEV_TARGET, *NEIGHBOR_SYMS):
            od = state.order_depths.get(sym)
            if od is None:
                continue
            b, a = _touch(od)
            if b is None or a is None:
                continue
            mids[sym] = 0.5 * (b + a)

        ub, ua = _touch(und)
        vb, va = _touch(vev)
        if ub is None or ua is None or vb is None or va is None:
            td["_iv"] = list(iv_hist)
            return result, 0, json.dumps(td)

        s_mid = 0.5 * (ub + ua)
        c_mid = 0.5 * (vb + va)
        mids[VEV_TARGET] = c_mid

        iv_now = implied_vol(s_mid, STRIKE, T_YEAR, c_mid)
        smile_ok = cheap_on_smile(s_mid, mids, STRIKE)

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

        if iv_now is not None and len(iv_hist) >= 51 and spread_ok:
            if q > 0 and z > -Z_EXIT:
                avail = vev.buy_orders.get(vb, 0)
                dq = min(q, ORDER_Q, avail)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, vb, -dq))
            elif q >= 0 and z < -Z_ENTRY and (smile_ok is not False):
                # smile_ok None => allow (same as v5 when neighbors missing)
                avail = abs(vev.sell_orders.get(va, 0))
                room = lim - q
                dq = min(ORDER_Q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, va, dq))

        result[VEV_TARGET] = orders_v

        td["_iv"] = list(iv_hist)[-ROLL_WIN - 50 :]
        td["_z"] = z
        td["_smile_ok"] = smile_ok
        return result, 0, json.dumps(td)
