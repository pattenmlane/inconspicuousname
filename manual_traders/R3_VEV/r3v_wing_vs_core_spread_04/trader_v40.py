"""
v40: v39 + joint "tight two-leg" regime from round3work/vouchers_final_strategy/STRATEGY.txt.

When VEV_5200 and VEV_5300 both have top-of-book spread <= TIGHT_S5200_S5300_TH (2 ticks, same
as the tape analysis), short-horizon extract mid is more favorable on average; when either
book is wide, execution noise dominates. We scale MAX_LEG and extract hedge: full size in the
tight joint regime, reduced in the wide regime (Sonic / inclineGod book-state gating).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from datamodel import Order, OrderDepth, TradingState
from scipy.optimize import brentq
from scipy.stats import norm

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"

LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in VOUCHERS},
}

RICH_THR = 0.010
CHEAP_THR = -0.022
MAX_LEG = 12
HEDGE_FRAC = 0.24
HYDRO_FRAC = 0.04
MIN_SPREAD_SKIP = 6
MIN_TIME_VALUE_FRAC = 0.045
SHOCK_DS_ABS = 2.0
SHOCK_CHEAP_SHIFT = -0.004

# Joint gate: both core vouchers tight (replicate analyze_vev_5200_5300_tight_gate_r3 TH=2)
TIGHT_S5200_S5300_TH = 2
# Wide-book regime: half clip, lighter hedge (still allow some flow; full stop would be 0)
LOOSE_MAX_LEG_FRAC = 0.5
LOOSE_HEDGE_FRAC = 0.12

WING_KS = {4000, 4500, 5400, 5500}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ref_S_at_open(day: int) -> float | None:
    p = _repo_root() / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv"
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 16:
                continue
            try:
                d_col, ts = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if d_col == day and ts == 0 and parts[2] == U:
                return float(parts[15])
    return None


def _infer_csv_day(S_mid: float, td: dict[str, Any]) -> int:
    if "csv_day" in td:
        return int(td["csv_day"])
    best_d, best_err = 0, 1e18
    for d in (0, 1, 2):
        ref = _ref_S_at_open(d)
        if ref is None:
            continue
        e = abs(S_mid - ref)
        if e < best_err:
            best_err, best_d = e, d
    if best_err < 15.0:
        td["csv_day"] = best_d
        return best_d
    td["csv_day"] = 0
    return 0


def _intraday_progress(ts: int) -> float:
    return (int(ts) // 100) / 10_000.0


def _dte_eff(csv_day: int, ts: int) -> float:
    return max(float(8 - int(csv_day)) - _intraday_progress(ts), 1e-6)


def _t_years(csv_day: int, ts: int) -> float:
    return _dte_eff(csv_day, ts) / 365.0


def _mid(depth: OrderDepth) -> tuple[float, int, int] | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    return (bb + ba) / 2.0, bb, ba


def _spread(depth: OrderDepth) -> int | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return min(depth.sell_orders) - max(depth.buy_orders)


def _joint_tight_regime(
    state: TradingState,
) -> tuple[bool, int | None, int | None]:
    """Tight iff both 5200 and 5300 BBO spreads exist and are <= TIGHT_S5200_S5300_TH."""
    d5 = state.order_depths.get(VEV_5200)
    d3 = state.order_depths.get(VEV_5300)
    s5 = _spread(d5) if d5 else None
    s3 = _spread(d3) if d3 else None
    if s5 is None or s3 is None:
        return False, s5, s3
    return (s5 <= TIGHT_S5200_S5300_TH and s3 <= TIGHT_S5200_S5300_TH), s5, s3


def bs_call(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def implied_vol(mid: float, S: float, K: float, T: float) -> float | None:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-6 or mid >= S - 1e-6 or S <= 0 or K <= 0 or T <= 0:
        return None

    def f(sig: float) -> float:
        return bs_call(S, K, T, sig, 0.0) - mid

    try:
        if f(1e-4) > 0 or f(12.0) < 0:
            return None
        return float(brentq(f, 1e-4, 12.0, xtol=1e-7, rtol=1e-7))
    except ValueError:
        return None


def bs_delta(mid: float, S: float, K: float, T: float) -> float | None:
    iv = implied_vol(mid, S, K, T)
    if iv is None or iv <= 0:
        return None
    v = iv * math.sqrt(T)
    if v <= 1e-12:
        return None
    d1 = (math.log(S / K) + 0.5 * iv * iv * T) / v
    return float(norm.cdf(d1))


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        orders: dict[str, list[Order]] = {p: [] for p in LIMITS}

        du = state.order_depths.get(U)
        if du is None:
            return orders, 0, json.dumps(td)
        mu = _mid(du)
        if mu is None:
            return orders, 0, json.dumps(td)
        S, _, _ = mu
        prev_s = float(td.get("prev_S", S))
        dS = S - prev_s
        td["prev_S"] = S
        csv_day = _infer_csv_day(S, td)
        ts = int(state.timestamp)
        T = _t_years(csv_day, ts)
        if T <= 0:
            return orders, 0, json.dumps(td)

        joint_tight, _s5200, _s5300 = _joint_tight_regime(state)
        max_leg_eff = MAX_LEG if joint_tight else max(1, int(round(LOOSE_MAX_LEG_FRAC * MAX_LEG)))
        hedge_fr_eff = HEDGE_FRAC if joint_tight else LOOSE_HEDGE_FRAC

        sqrtT = math.sqrt(T)
        xs: list[float] = []
        ys: list[float] = []
        strikes_data: list[tuple[str, int, float, float, float | None, float]] = []

        for v in VOUCHERS:
            d = state.order_depths.get(v)
            if d is None:
                continue
            m = _mid(d)
            if m is None:
                continue
            mid, _, _ = m
            K = int(v.split("_")[1])
            intrinsic = max(S - K, 0.0)
            tv = mid - intrinsic
            if mid <= 0 or tv / mid < MIN_TIME_VALUE_FRAC:
                continue
            iv = implied_vol(mid, S, K, T)
            if iv is None:
                continue
            m_t = math.log(K / S) / sqrtT
            xs.append(m_t)
            ys.append(iv)
            strikes_data.append((v, K, mid, iv, bs_delta(mid, S, K, T), tv / mid))

        if len(xs) < 6:
            return orders, 0, json.dumps(td)

        coeff = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 2)
        a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])

        d_net = 0.0
        for v, K, mid, iv, delta, tv_frac in strikes_data:
            if delta is None or K not in WING_KS:
                continue
            m_t = math.log(K / S) / sqrtT
            resid = iv - (a * m_t * m_t + b * m_t + c)
            sp = _spread(state.order_depths[v])
            if sp is None or sp > MIN_SPREAD_SKIP:
                continue
            pos = state.position.get(v, 0)
            lim = LIMITS[v]
            cheap_thr_eff = CHEAP_THR
            if abs(dS) >= SHOCK_DS_ABS:
                cheap_thr_eff = CHEAP_THR + SHOCK_CHEAP_SHIFT

            qty = 0
            if resid > RICH_THR and pos > -lim + 1:
                qty = -min(max_leg_eff, pos + lim)
            elif resid < cheap_thr_eff and pos < lim - 1 and tv_frac >= MIN_TIME_VALUE_FRAC:
                qty = min(max_leg_eff, lim - pos)
            if qty == 0:
                continue
            d = state.order_depths[v]
            if qty > 0 and d.sell_orders:
                orders[v].append(Order(v, min(d.sell_orders), qty))
            elif qty < 0 and d.buy_orders:
                orders[v].append(Order(v, max(d.buy_orders), qty))
            d_net += delta * qty

        pu = state.position.get(U, 0)
        u_lim = LIMITS[U]
        hedge_q = -int(round(hedge_fr_eff * d_net))
        hedge_q = max(-u_lim - pu, min(u_lim - pu, hedge_q))
        if hedge_q != 0 and du.buy_orders and du.sell_orders:
            if hedge_q > 0:
                orders[U].append(Order(U, min(du.sell_orders), hedge_q))
            else:
                orders[U].append(Order(U, max(du.buy_orders), hedge_q))

        dh = state.position.get(H, 0)
        h_lim = LIMITS[H]
        tilt = -int(round(HYDRO_FRAC * hedge_q))
        tilt = max(-h_lim - dh, min(h_lim - dh, tilt))
        dh_dep = state.order_depths.get(H)
        if tilt != 0 and dh_dep and dh_dep.buy_orders and dh_dep.sell_orders:
            if tilt > 0:
                orders[H].append(Order(H, min(dh_dep.sell_orders), tilt))
            else:
                orders[H].append(Order(H, max(dh_dep.buy_orders), tilt))

        return orders, 0, json.dumps(td)
