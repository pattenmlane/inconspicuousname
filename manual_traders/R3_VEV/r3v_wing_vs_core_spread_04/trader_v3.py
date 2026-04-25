"""
v3: Curvature fade — sell rich wings only (no cheap-wing longs; those hurt in tape).
Parent: v0. Sweep: lower RICH_THR, HEDGE_FRAC 0.72, skip wide spreads.
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

LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in VOUCHERS},
}

RICH_THR = 0.010
MAX_LEG = 16
HEDGE_FRAC = 0.72
HYDRO_FRAC = 0.06
MIN_SPREAD_SKIP = 7

WING_KS = {4000, 4500, 5400, 5500, 6000, 6500}


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
        csv_day = _infer_csv_day(S, td)
        ts = int(state.timestamp)
        T = _t_years(csv_day, ts)
        if T <= 0:
            return orders, 0, json.dumps(td)

        sqrtT = math.sqrt(T)
        xs: list[float] = []
        ys: list[float] = []
        strikes_data: list[tuple[str, int, float, float, float | None]] = []

        for v in VOUCHERS:
            d = state.order_depths.get(v)
            if d is None:
                continue
            m = _mid(d)
            if m is None:
                continue
            mid, _, _ = m
            K = int(v.split("_")[1])
            iv = implied_vol(mid, S, K, T)
            if iv is None:
                continue
            m_t = math.log(K / S) / sqrtT
            xs.append(m_t)
            ys.append(iv)
            strikes_data.append((v, K, mid, iv, bs_delta(mid, S, K, T)))

        if len(xs) < 6:
            return orders, 0, json.dumps(td)

        coeff = np.polyfit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), 2)
        a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])

        def iv_hat(m: float) -> float:
            return a * m * m + b * m + c

        d_net = 0.0
        for v, K, mid, iv, delta in strikes_data:
            if delta is None or K not in WING_KS:
                continue
            m_t = math.log(K / S) / sqrtT
            resid = iv - iv_hat(m_t)
            sp = _spread(state.order_depths[v])
            if sp is None or sp > MIN_SPREAD_SKIP:
                continue
            pos = state.position.get(v, 0)
            lim = LIMITS[v]
            if resid <= RICH_THR or pos <= -lim + 1:
                continue
            qty = -min(MAX_LEG, pos + lim)
            if qty == 0:
                continue
            d = state.order_depths[v]
            if d.buy_orders:
                orders[v].append(Order(v, max(d.buy_orders), qty))
                d_net += delta * qty

        pu = state.position.get(U, 0)
        u_lim = LIMITS[U]
        hedge_q = -int(round(HEDGE_FRAC * d_net))
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
