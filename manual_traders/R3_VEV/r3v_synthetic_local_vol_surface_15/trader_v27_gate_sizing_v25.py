"""
v27: Parent v25 (family12 uniform, trade only VEV_5000/5100 — 5200 stays off).

**Joint tight two-leg gate** (round3work/vouchers_final_strategy/STRATEGY.txt): at each tick,
if VEV_5200 and VEV_5300 both have top-of-book spread ≤ 2, treat the book as a **risk-on**
regime and slightly **tighten** the residual hurdle and **increase** clip size. When either
is wide, **widen** the hurdle and **shrink** size — do not re-introduce 5200 (v25 tape
analysis showed 5200 far noisier per half-spread than 5000/5100).

No VELVETFRUIT_EXTRACT orders: mid-forward edge from STRATEGY does not survive bb−1/ba+1
execution in our tests (see v26).
"""
from __future__ import annotations

import json
import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.stats import norm

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
V_GATE_A = "VEV_5200"
V_GATE_B = "VEV_5300"

REFIT_EVERY_TICKS = 25
RESID_HURDLE_MULT = 1.28
SPLINE_SMOOTH = 0.08
MAX_ORDER_PER_VOUCHER = 16
WARMUP_TICKS = 30
POS_VEV = 300
IV_NEWTON_ITERS = 10
ALLOWED_STRIKES = {5000, 5100}
VEGA_HURDLE_COEF = 0.0010
WING_STRIKES = {4000, 4500, 6000, 6500}
WING_SPLINE_WEIGHT = 0.58
KNOT_FAMILY = "uniform"

TIGHT_SPREAD_TH = 2
HURDLE_MULT_TIGHT = 0.97
HURDLE_MULT_WIDE = 1.05
ORDER_MULT_TIGHT = 1.1
ORDER_MULT_WIDE = 0.9


def _dte_open_from_day(day: int) -> float:
    return float(8 - int(day))


def _t_years(day: int, ts: int) -> float:
    dte_eff = max(_dte_open_from_day(day) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte_eff / 365.0


def _best_bid_ask(depth: OrderDepth) -> tuple[int | None, int | None]:
    buys = depth.buy_orders
    sells = depth.sell_orders
    if not buys or not sells:
        return None, None
    bb = max(buys.keys())
    ba = min(abs(p) for p in sells.keys())
    return bb, ba


def _spread_width(depth: OrderDepth) -> int | None:
    bb, ba = _best_bid_ask(depth)
    if bb is None or ba is None or ba <= bb:
        return None
    return int(ba - bb)


def _bs_call(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _vega(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0 or sig <= 1e-12:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    return S * norm.pdf(d1) * math.sqrt(T)


def _iv_newton(mid: float, S: float, K: float, T: float, guess: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-6 or mid >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    sig = max(min(guess, 6.0), 0.04)
    for _ in range(IV_NEWTON_ITERS):
        pr = _bs_call(S, K, T, sig, r) - mid
        if abs(pr) < 1e-4:
            break
        veg = _vega(S, K, T, sig, r)
        if veg < 1e-8:
            return float("nan")
        sig -= pr / veg
        sig = max(min(sig, 8.0), 0.03)
    if abs(_bs_call(S, K, T, sig, r) - mid) > 0.05:
        return float("nan")
    return sig


def _fit_base_iv_spline(
    xs: np.ndarray, ivs: np.ndarray, weights: np.ndarray, s: float
) -> UnivariateSpline | None:
    if xs.size < 4:
        return None
    order = np.argsort(xs)
    xo = xs[order]
    vo = ivs[order]
    wo = weights[order]
    k = 3 if xo.size > 3 else min(2, xo.size - 1)
    if k < 1:
        return None
    try:
        return UnivariateSpline(xo, vo, w=wo, k=k, s=s)
    except Exception:
        return None


def _pseudo_knot_grid(xmin: float, xmax: float, family: str) -> np.ndarray:
    if family == "uniform":
        return np.linspace(xmin, xmax, 12)
    core = np.array([-0.08, -0.05, -0.03, -0.015, 0.0, 0.015, 0.03, 0.05, 0.08])
    wings = np.array([xmin, xmin * 0.6, xmax * 0.6, xmax])
    return np.unique(np.clip(np.concatenate([wings, core]), xmin, xmax))


def _fit_family_spline(
    xs: np.ndarray, ivs: np.ndarray, weights: np.ndarray, s: float, family: str
) -> CubicSpline | None:
    base = _fit_base_iv_spline(xs, ivs, weights, s)
    if base is None:
        return None
    order = np.argsort(xs)
    xo = xs[order]
    xmin, xmax = float(xo[0]), float(xo[-1])
    grid = _pseudo_knot_grid(xmin, xmax, family)
    try:
        yg = np.asarray(base(grid), dtype=float)
        return CubicSpline(grid, yg, bc_type="natural", extrapolate=True)
    except Exception:
        return None


class Trader:
    def run(self, state: TradingState):
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}
        ts = int(getattr(state, "timestamp", 0))
        tick = ts // 100

        raw = getattr(state, "traderData", None) or ""
        try:
            store: dict[str, Any] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            store = {}
        if not isinstance(store, dict):
            store = {}

        csv_day = int(getattr(state, "backtest_day", 0))
        tick_key = "tick_count"
        store[tick_key] = int(store.get(tick_key, 0)) + 1
        n_tick = int(store[tick_key])

        iv_prev: dict[str, float] = store.get("iv_prev", {})
        if not isinstance(iv_prev, dict):
            iv_prev = {}
        iv_prev = {str(k): float(v) for k, v in iv_prev.items() if isinstance(v, (int, float))}

        depth_u = depths.get(U)
        if depth_u is None:
            store["iv_prev"] = iv_prev
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ubb, uba = _best_bid_ask(depth_u)
        if ubb is None or uba is None or uba <= ubb:
            store["iv_prev"] = iv_prev
            return {}, 0, json.dumps(store, separators=(",", ":"))

        S = 0.5 * float(ubb + uba)
        T = _t_years(csv_day, ts)

        d52 = depths.get(V_GATE_A)
        d53 = depths.get(V_GATE_B)
        s1 = _spread_width(d52) if d52 is not None else None
        s2 = _spread_width(d53) if d53 is not None else None
        joint_tight = (
            s1 is not None
            and s2 is not None
            and s1 <= TIGHT_SPREAD_TH
            and s2 <= TIGHT_SPREAD_TH
        )
        hmul = HURDLE_MULT_TIGHT if joint_tight else HURDLE_MULT_WIDE
        omul = ORDER_MULT_TIGHT if joint_tight else ORDER_MULT_WIDE

        if tick < WARMUP_TICKS:
            store["iv_prev"] = iv_prev
            return {}, 0, json.dumps(store, separators=(",", ":"))

        knots_x: list[float] = []
        knots_iv: list[float] = []
        knot_w: list[float] = []
        mids: dict[str, tuple[float, float, int, int]] = {}

        for v in VOUCHERS:
            d = depths.get(v)
            if d is None:
                continue
            bb, ba = _best_bid_ask(d)
            if bb is None or ba is None or ba <= bb:
                continue
            mid = 0.5 * (bb + ba)
            k = int(v.split("_")[1])
            g = float(iv_prev.get(v, 0.45))
            iv = _iv_newton(mid, S, k, T, g, 0.0)
            if not math.isfinite(iv):
                continue
            iv_prev[v] = iv
            x = math.log(max(k / S, 1e-6))
            knots_x.append(x)
            knots_iv.append(iv)
            knot_w.append(WING_SPLINE_WEIGHT if k in WING_STRIKES else 1.0)
            mids[v] = (mid, 0.5 * (ba - bb), bb, ba)

        _ = (n_tick % REFIT_EVERY_TICKS == 0)
        iv_mean = float(np.mean(knots_iv)) if knots_iv else 0.45

        spl: CubicSpline | None = None
        if len(knots_iv) >= 4:
            spl = _fit_family_spline(
                np.asarray(knots_x, float),
                np.asarray(knots_iv, float),
                np.asarray(knot_w, float),
                SPLINE_SMOOTH,
                KNOT_FAMILY,
            )

        def sigma_hat(x: float) -> float:
            if spl is not None:
                try:
                    v = float(spl(float(x)))
                    return max(min(v, 7.5), 0.03)
                except Exception:
                    pass
            return iv_mean

        orders_out: dict[str, list[Order]] = {}

        for v, (mid, hsp, bb, ba) in mids.items():
            k = int(v.split("_")[1])
            if k not in ALLOWED_STRIKES:
                continue
            x = math.log(max(k / S, 1e-6))
            sig_hat = sigma_hat(x)
            theo = _bs_call(S, k, T, sig_hat, 0.0)
            resid = mid - theo
            pos_v = int(pos.get(v, 0))
            veg = _vega(S, k, T, sig_hat, 0.0)
            hurdle = float(hmul) * (RESID_HURDLE_MULT * max(hsp, 0.5) + VEGA_HURDLE_COEF * veg)
            if abs(resid) < hurdle:
                continue
            qscale = min(MAX_ORDER_PER_VOUCHER, max(3, int(2.0 + abs(resid) / max(hsp, 1.0))))
            if veg < 5.0:
                qscale = max(3, qscale // 2)
            qscale = int(max(1, min(MAX_ORDER_PER_VOUCHER, int(qscale * omul + 0.5))))

            if resid > 0 and pos_v > -POS_VEV + qscale:
                px = max(int(bb) - 1, 0)
                q = min(qscale, pos_v + POS_VEV)
                if q > 0:
                    orders_out.setdefault(v, []).append(Order(v, px, -q))
            elif resid < 0 and pos_v < POS_VEV - qscale:
                px = int(ba) + 1
                q = min(qscale, POS_VEV - pos_v)
                if q > 0:
                    orders_out.setdefault(v, []).append(Order(v, px, q))

        store["iv_prev"] = iv_prev
        return orders_out, 0, json.dumps(store, separators=(",", ":"))
