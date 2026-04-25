"""
v14: v10 log-IV rolling z + tape-informed dynamic spread cap on entry.

Tape analysis (iter 13): in cheap-vol tails (log-IV z very negative), VEV_5000
top-of-book spreads are wider (~5 ticks) and extract often mean-reverts down
while the option drifts up. v10 used MAX_SPREAD=2 everywhere, which may skip
those moments. Here we keep v10’s core (ROLL_WIN, Z_ENTRY, Z_EXIT, ORDER_Q)
but allow a **relaxed** spread cap only for **new long entry** when recent
extract mid dropped over a short lookback (conditional liquidity).

Exit trades (flattening long when z mean-reverts) still use the tight cap so
we do not pay extra edge on unwinds.

Parameters: base spread 2; relaxed **6**; lookback **12** ticks; drop threshold
**2.0** (small grid on Round 3 days 0–2, worse fills — best of tested slice).

IV: BS implied vol from mids (r=0, T=7/365). Greek context unchanged.

TTE: round3description — ROUND_3 day 0/1/2 ~ 8d/7d/6d at tape start.
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
IV_FLOOR = 0.01

ROLL_WIN = 400
Z_ENTRY = 2.6
Z_EXIT = 0.07
MAX_SPREAD_BASE = 2
MAX_SPREAD_RELAXED = 6
S_LOOKBACK = 12
DROP_THRESH = 2.0
ORDER_Q = 6
MIN_LOG_STD = 1e-4


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

        log_hist: deque[float] = deque(
            (float(x) for x in td.get("_log_iv", [])), maxlen=ROLL_WIN + 100
        )
        s_hist: deque[float] = deque(
            (float(x) for x in td.get("_s_mid", [])), maxlen=max(S_LOOKBACK + 5, 20)
        )
        result: dict[str, list[Order]] = {p: [] for p in LIMITS}

        und = state.order_depths.get(UNDER)
        vev = state.order_depths.get(VEV_TARGET)
        pos = state.position
        if und is None or vev is None:
            td["_log_iv"] = list(log_hist)
            td["_s_mid"] = list(s_hist)
            return result, 0, json.dumps(td)

        ub, ua = _touch(und)
        vb, va = _touch(vev)
        if ub is None or ua is None or vb is None or va is None:
            td["_log_iv"] = list(log_hist)
            td["_s_mid"] = list(s_hist)
            return result, 0, json.dumps(td)

        s_mid = 0.5 * (ub + ua)
        c_mid = 0.5 * (vb + va)
        iv_now = implied_vol(s_mid, STRIKE, T_YEAR, c_mid)
        log_now = math.log(max(iv_now, IV_FLOOR)) if iv_now is not None else None

        z = 0.0
        if log_now is not None and len(log_hist) >= 50:
            mu = sum(log_hist) / len(log_hist)
            var = sum((x - mu) ** 2 for x in log_hist) / max(len(log_hist), 1)
            sig = math.sqrt(var)
            if sig < MIN_LOG_STD:
                sig = MIN_LOG_STD
            z = (log_now - mu) / sig

        if log_now is not None:
            log_hist.append(log_now)

        s_hist.append(s_mid)
        recent_drop = False
        if len(s_hist) > S_LOOKBACK:
            s0 = s_hist[-(S_LOOKBACK + 1)]
            s1 = s_hist[-1]
            if s0 - s1 >= DROP_THRESH:
                recent_drop = True
        sp = va - vb
        spread_ok_exit = sp <= MAX_SPREAD_BASE
        spread_ok_entry = sp <= (MAX_SPREAD_RELAXED if recent_drop else MAX_SPREAD_BASE)
        q = pos.get(VEV_TARGET, 0)
        lim = LIMITS[VEV_TARGET]
        orders_v: list[Order] = []

        if log_now is not None and len(log_hist) >= 51:
            if q > 0 and z > -Z_EXIT and spread_ok_exit:
                avail = vev.buy_orders.get(vb, 0)
                dq = min(q, ORDER_Q, avail)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, vb, -dq))
            elif q >= 0 and z < -Z_ENTRY and spread_ok_entry:
                avail = abs(vev.sell_orders.get(va, 0))
                room = lim - q
                dq = min(ORDER_Q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, va, dq))

        result[VEV_TARGET] = orders_v

        td["_log_iv"] = list(log_hist)[-ROLL_WIN - 50 :]
        td["_s_mid"] = list(s_hist)[-S_LOOKBACK - 5 :]
        td["_z"] = z
        td["_iv"] = float(iv_now) if iv_now is not None else 0.0
        td["_sp_cap"] = MAX_SPREAD_RELAXED if recent_drop else MAX_SPREAD_BASE
        return result, 0, json.dumps(td)
