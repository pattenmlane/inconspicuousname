"""
Shared Round-3 helpers: DTE/TTE (round3work/round3description + intraday wind),
Black–Scholes call, bisection IV, piecewise-linear IV in strike between knots.
"""
from __future__ import annotations

import json
import math
from typing import Any

try:
    from scipy.stats import norm as _norm
except ImportError:  # pragma: no cover
    _norm = None  # type: ignore


def dte_from_csv_day(day: int) -> int:
    """Calendar DTE at session open for historical CSV day index (see round3description)."""
    return 8 - int(day)


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years_effective(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def _ncdf(x: float) -> float:
    if _norm is not None:
        return float(_norm.cdf(x))
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return float(_ncdf(d1))


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return float(S * _npdf(d1) * math.sqrt(T))


def bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black–Scholes call gamma d²C/dS² (per unit S), r=0 for Prosperity round-3."""
    if T <= 0 or sigma <= 1e-12 or S <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return float(_npdf(d1) / (S * v))


def bs_call_theta(
    S: float, K: float, T: float, sigma: float, r: float = 0.0
) -> float:
    """
    d(option_price)/dT (per year), Black-Scholes call. T in years, r=0 for Prosperity round-3.
    """
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    term1 = -0.5 * S * _npdf(d1) * sigma / math.sqrt(T)
    term2 = -r * K * math.exp(-r * T) * _ncdf(d2)
    return float(term1 + term2)


def implied_vol_bisect(
    market: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    lo: float = 1e-4,
    hi: float = 12.0,
    it: int = 40,
) -> float | None:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return None
    if market >= S - 1e-9:
        return None

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    try:
        from scipy.optimize import brentq

        return float(brentq(f, lo, hi, xtol=1e-5, rtol=1e-5, maxiter=it))
    except Exception:
        pass

    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        return None
    a, b = lo, hi
    fa, fb = flo, fhi
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < 1e-6:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


# Default interior knots (strike); linear in K between consecutive knots; flat outside range.
DEFAULT_KNOT_STRIKES = (5000, 5200, 5400)


def pwl_iv_strike(
    K: float,
    knot_strikes: tuple[int, ...],
    knot_ivs: tuple[float, ...],
) -> float:
    """Piecewise linear sigma(K) with flat extrapolation outside [knot_strikes[0], knot_strikes[-1]]."""
    ks = knot_strikes
    vs = knot_ivs
    if K <= ks[0]:
        return float(vs[0])
    if K >= ks[-1]:
        return float(vs[-1])
    for i in range(len(ks) - 1):
        if ks[i] <= K <= ks[i + 1]:
            t = (K - ks[i]) / (ks[i + 1] - ks[i])
            return float(vs[i] * (1 - t) + vs[i + 1] * t)
    return float(vs[-1])


def fit_knot_ivs_least_squares(
    strikes: list[int],
    ivs: list[float],
    knot_strikes: tuple[int, ...] = DEFAULT_KNOT_STRIKES,
) -> tuple[float, ...] | None:
    """Initial knot IVs from observed IVs at strikes nearest each knot."""
    valid = [(k, v) for k, v in zip(strikes, ivs) if math.isfinite(v) and v > 0]
    if len(valid) < 2:
        return None
    byk = {k: v for k, v in valid}

    def clip_iv(x: float) -> float:
        return max(0.03, min(5.0, x))

    def nearest_iv(target_k: int) -> float:
        best_k, best_d = None, 10**9
        for k, v in valid:
            d = abs(k - target_k)
            if d < best_d:
                best_d, best_k = d, k
        return float(byk[best_k]) if best_k is not None else float(valid[0][1])

    w1 = clip_iv(nearest_iv(knot_strikes[0]))
    w2 = clip_iv(nearest_iv(knot_strikes[1]))
    w3 = clip_iv(nearest_iv(knot_strikes[2]))
    return (w1, w2, w3)


def refine_knot_ivs_gauss_newton(
    strikes: list[int],
    ivs: list[float],
    init: tuple[float, float, float],
    knot_strikes: tuple[int, ...] = DEFAULT_KNOT_STRIKES,
    steps: int = 12,
    lr: float = 0.35,
) -> tuple[float, float, float]:
    w = list(init)
    ks = list(knot_strikes)
    for _ in range(steps):
        grad = [0.0, 0.0, 0.0]
        loss_w = 0.0
        for K, iv_t in zip(strikes, ivs):
            if not math.isfinite(iv_t) or iv_t <= 0:
                continue
            pred = pwl_iv_strike(float(K), knot_strikes, (w[0], w[1], w[2]))
            r = pred - iv_t
            loss_w += r * r
            # d pred / d w_j: which segment?
            Kf = float(K)
            if Kf <= ks[0]:
                grad[0] += 2 * r * 1.0
            elif Kf >= ks[2]:
                grad[2] += 2 * r * 1.0
            elif ks[0] < Kf < ks[1]:
                t = (Kf - ks[0]) / (ks[1] - ks[0])
                grad[0] += 2 * r * (1 - t)
                grad[1] += 2 * r * t
            else:
                t = (Kf - ks[1]) / (ks[2] - ks[1])
                grad[1] += 2 * r * (1 - t)
                grad[2] += 2 * r * t
        for j in range(3):
            w[j] = max(0.03, min(5.0, w[j] - lr * grad[j]))
    return (w[0], w[1], w[2])


def book_walls(depth: Any) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """(bid_wall, ask_wall, best_bid, best_ask, wall_mid)."""
    try:
        raw_buys = getattr(depth, "buy_orders", None) or {}
        raw_sells = getattr(depth, "sell_orders", None) or {}
        buys = sorted((int(bp), abs(int(bv))) for bp, bv in raw_buys.items())
        sells = sorted((int(sp), abs(int(sv))) for sp, sv in raw_sells.items())
    except (TypeError, ValueError, KeyError):
        return None, None, None, None, None
    if not buys and not sells:
        return None, None, None, None, None
    bid_prices = [p for p, _ in buys]
    ask_prices = [p for p, _ in sells]
    bid_wall = min(bid_prices)
    ask_wall = max(ask_prices)
    best_bid = max(bid_prices)
    best_ask = min(ask_prices)
    wm = 0.5 * (float(bid_wall) + float(ask_wall))
    return float(bid_wall), float(ask_wall), float(best_bid), float(best_ask), wm


def synthetic_walls(
    bid_wall: float | None,
    ask_wall: float | None,
    best_bid: float | None,
    best_ask: float | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    if bid_wall is None and ask_wall is not None:
        bid_wall = float(ask_wall) - 1.0
        best_bid = bid_wall
        wm = float(ask_wall) - 0.5
        return bid_wall, ask_wall, wm, best_bid, float(ask_wall)
    if ask_wall is None and bid_wall is not None:
        ask_wall = float(bid_wall) + 1.0
        best_ask = ask_wall
        wm = float(bid_wall) + 0.5
        return bid_wall, ask_wall, wm, float(bid_wall), best_ask
    if bid_wall is not None and ask_wall is not None:
        wm = (float(bid_wall) + float(ask_wall)) / 2.0
        return bid_wall, ask_wall, wm, best_bid, best_ask
    return None, None, None, best_bid, best_ask


def parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def infer_csv_day_from_smile(
    S: float,
    strike_mids: dict[int, float],
    timestamp: int,
    strikes: tuple[int, ...],
    probe_strikes: tuple[int, ...] = (5100, 5200, 5300),
) -> int:
    """
    Pick csv_day in {0,1,2} minimizing cross-strike IV dispersion at that T
    (IV from mid via bisection; r=0). Uses a few ATM-ish strikes only for speed.
    """
    if S <= 0:
        return 0
    best_d, best_score = 0, float("inf")
    for d in (0, 1, 2):
        T = t_years_effective(d, timestamp)
        ivs: list[float] = []
        for K in probe_strikes:
            mid = strike_mids.get(K)
            if mid is None:
                continue
            iv = implied_vol_bisect(float(mid), S, float(K), T)
            if iv is not None and math.isfinite(iv):
                ivs.append(iv)
        if len(ivs) < 2:
            continue
        med = sorted(ivs)[len(ivs) // 2]
        score = sum((v - med) ** 2 for v in ivs)
        if score < best_score:
            best_score, best_d = score, d
    return int(best_d)
