"""
v18: v17 + **joint 5200/5300 book state** (STRATEGY.txt observability layer).

`round3work/vouchers_final_strategy/STRATEGY.txt`: when **VEV_5200** and **VEV_5300** both
have top-of-book spread ≤ 2, forward extract-mid edge is cleaner (tight vs not-tight split in
`outputs/r3_tight_spread_summary.txt`). v18 tracks `joint_tight` and per-leg spreads in
`traderData` for debugging / future gating, while keeping **v17** execution unchanged: a
parameter sweep of entry clip boost when (drawdown ∧ joint_tight) did not move **Round 3
days 0–2** worse-fill PnL vs v17 (still 19,780 total).

Hydro not traded. T=7/365, r=0. TTE: round3description.
"""

from __future__ import annotations

import json
import math
from collections import deque
from datamodel import Order, OrderDepth, TradingState

VEV_TARGET = "VEV_5000"
VEV_GATE_A = "VEV_5200"
VEV_GATE_B = "VEV_5300"
UNDER = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"

# Joint gate: both neighbors tight (see STRATEGY.txt, TH=2 on Round 3 tape)
GATE_MAX_SPREAD = 2

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
S_LOOKBACK = 10
DROP_THRESH = 1.0
VEGA_TARGET = 1100.0
Q_MIN = 4
Q_MAX = 10

MIN_LOG_STD = 1e-4


def _N_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(spot: float, strike: float, t: float, vol: float) -> float:
    if t <= 0 or vol <= 0:
        return max(spot - strike, 0.0)
    sig_rt = vol * math.sqrt(t)
    if sig_rt < 1e-12:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / sig_rt
    d2 = d1 - sig_rt
    return spot * _N_cdf(d1) - strike * _N_cdf(d2)


def bs_call_vega(spot: float, strike: float, t: float, vol: float) -> float:
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    sig_rt = vol * math.sqrt(t)
    if sig_rt < 1e-12:
        return 0.0
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / sig_rt
    return spot * _phi(d1) * math.sqrt(t)


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


def clip_vega(vega_per_unit: float) -> int:
    if vega_per_unit < 1e-8:
        return (Q_MIN + Q_MAX) // 2
    q = int(round(VEGA_TARGET / vega_per_unit))
    return max(Q_MIN, min(Q_MAX, q))


def _touch(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


def _book_spread(depth: OrderDepth) -> int | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return min(depth.sell_orders) - max(depth.buy_orders)


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
        d52 = state.order_depths.get(VEV_GATE_A)
        d53 = state.order_depths.get(VEV_GATE_B)
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

        sp52 = _book_spread(d52) if d52 is not None else None
        sp53 = _book_spread(d53) if d53 is not None else None
        joint_tight = (
            sp52 is not None
            and sp53 is not None
            and sp52 <= GATE_MAX_SPREAD
            and sp53 <= GATE_MAX_SPREAD
        )

        s_mid = 0.5 * (ub + ua)
        c_mid = 0.5 * (vb + va)
        iv_now = implied_vol(s_mid, STRIKE, T_YEAR, c_mid)
        log_now = math.log(max(iv_now, IV_FLOOR)) if iv_now is not None else None
        vega_u = bs_call_vega(s_mid, STRIKE, T_YEAR, iv_now) if iv_now is not None else 0.0
        order_q = clip_vega(vega_u)

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
        # Match v17: relaxed V5000 cap on extract drawdown.
        spread_ok_entry = sp <= (MAX_SPREAD_RELAXED if recent_drop else MAX_SPREAD_BASE)

        q5 = pos.get(VEV_TARGET, 0)
        lim5 = LIMITS[VEV_TARGET]
        orders_v: list[Order] = []

        if log_now is not None and len(log_hist) >= 51:
            if q5 > 0 and z > -Z_EXIT and spread_ok_exit:
                avail = vev.buy_orders.get(vb, 0)
                dq = min(q5, order_q, avail)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, vb, -dq))
            elif q5 >= 0 and z < -Z_ENTRY and spread_ok_entry:
                avail = abs(vev.sell_orders.get(va, 0))
                room = lim5 - q5
                dq = min(order_q, avail, room)
                if dq > 0:
                    orders_v.append(Order(VEV_TARGET, va, dq))

        result[VEV_TARGET] = orders_v

        td["_log_iv"] = list(log_hist)[-ROLL_WIN - 50 :]
        td["_s_mid"] = list(s_hist)[-S_LOOKBACK - 5 :]
        td["_z"] = z
        td["_iv"] = float(iv_now) if iv_now is not None else 0.0
        td["_vega"] = vega_u
        td["_order_q"] = float(order_q)
        td["_sp_cap"] = float(MAX_SPREAD_RELAXED if recent_drop else MAX_SPREAD_BASE)
        td["_joint_tight"] = 1.0 if joint_tight else 0.0
        td["_s5200"] = float(sp52) if sp52 is not None else -1.0
        td["_s5300"] = float(sp53) if sp53 is not None else -1.0
        return result, 0, json.dumps(td)
