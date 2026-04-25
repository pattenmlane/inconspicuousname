"""
Iteration 10: parent v9. Same WLS smile + BS fair + |theta| scaling.

Spread microstructure: on ROUND_3 tape, top-of-book spread correlates ~0.68 with
|mid-fair|/vega (see spread_vegaedge_corr_iter10.json). Widen the theta-based gate
multiplier slightly on wide books so we are pickier when microstructure is noisy
(Microprice unchanged; this only scales the same v9 edge / edge-per-vega tests).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from datamodel import Order, OrderDepth, TradingState
from scipy.stats import norm

try:
    from prosperity4bt.constants import LIMITS
except ImportError:
    LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
    }

U = "VELVETFRUIT_EXTRACT"
GEL = "HYDROGEL_PACK"
VOUCHERS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
CORE_STRIKES = {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400"}

_PRIOR_PATH = Path(__file__).resolve().parent / "smile_wls_detail.json"
_EDGE_FRAC = 0.27
_EDGE_OVER_VEGA_MIN = 0.009
_MIN_VEGA = 0.05
_MIN_VEV_SPREAD = 2
_RIDGE_LAMBDA = 5e-4
_BUFFER_CAP = 400
_RECENTER_EVERY = 80
_PRIOR_ONLY_STEPS = 120
_GEL_SIZE = 22
_EXTRACT_MM = 10
_WARMUP_STEPS = 10
# Theta in price units / year at ~ATM can be 500–2000+ on this tape; scale so mult stays ~1.0–1.2
_THETA_REF = 1200.0
_THETA_EDGE_MULT = 0.15
# spr is often 2 (median); cap extra strictness for very wide books (p75 spread ~6 on tape)
_SPREAD_GATE_COEF = 0.08
_SPREAD_GATE_CAP = 1.0 + 0.5 * _SPREAD_GATE_COEF


def _load_prior() -> tuple[float, float, float]:
    try:
        d = json.loads(_PRIOR_PATH.read_text(encoding="utf-8"))
        c = d["wls_coeffs_high_to_low"]
        return float(c[0]), float(c[1]), float(c[2])
    except (OSError, KeyError, json.JSONDecodeError, TypeError, IndexError):
        return 0.14, -0.006, 0.235


_PA, _PB, _PC = _load_prior()


def dte_effective(csv_day: int, ts: int) -> float:
    return max(8.0 - float(csv_day) - (ts // 100) / 10000.0, 1e-6)


def t_years(csv_day: int, ts: int) -> float:
    return dte_effective(csv_day, ts) / 365.0


def bs_call(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _d1(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    v = sig * math.sqrt(T)
    return (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v


def vega(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0 or sig <= 1e-12:
        return 0.0
    d1 = _d1(S, K, T, sig, r)
    return float(S * norm.pdf(d1) * math.sqrt(T))


def call_theta(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    """dV/dT (r=0), in price per year. Negative for long calls; we use abs in gates."""
    if T <= 0 or sig <= 1e-12:
        return 0.0
    d1 = _d1(S, K, T, sig, r)
    v = sig * math.sqrt(T)
    return -S * norm.pdf(d1) * sig / (2.0 * v)


def theta_edge_multiplier(abs_theta: float) -> float:
    t = min(abs(abs_theta), 3.0 * _THETA_REF)
    return 1.0 + _THETA_EDGE_MULT * (t / (t + _THETA_REF))


def spread_gate_multiplier(spr: int) -> float:
    """1 on tight spread, up to small cap on wide spread (informed by tape corr spread vs |edge|/vega)."""
    w = (float(spr) - 2.0) / 6.0
    w = min(1.0, max(0.0, w))
    return min(_SPREAD_GATE_CAP, 1.0 + _SPREAD_GATE_COEF * w)


def implied_vol_newton(mid: float, S: float, K: float, T: float, r: float = 0.0, guess: float = 0.35) -> float | None:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-9 or mid >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return None
    sig = max(min(guess, 5.0), 1e-4)
    for _ in range(28):
        pr = bs_call(S, K, T, sig, r)
        vg = vega(S, K, T, sig, r)
        if vg < 1e-12:
            return None
        step = (pr - mid) / vg
        sig -= step
        if sig <= 1e-5:
            sig = 1e-5
        if abs(step) < 1e-7:
            return float(sig)
        if sig > 15.0:
            return None
    return float(sig) if 1e-5 < sig < 15.0 else None


def best_bid_ask(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if depth is None:
        return None, None
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


def sigma_from_abc(m: float, a: float, b: float, c: float) -> float:
    return max(a * m * m + b * m + c, 0.02)


def ridge_abc(xs: list[float], ys: list[float], ws: list[float], lam: float, pa: float, pb: float, pc: float) -> tuple[float, float, float]:
    if not xs:
        return pa, pb, pc
    g00 = g01 = g02 = g11 = g12 = g22 = 0.0
    r0 = r1 = r2 = 0.0
    for i, x in enumerate(xs):
        w = ws[i]
        x0, x1, x2 = x * x, x, 1.0
        g00 += w * x0 * x0
        g01 += w * x0 * x1
        g02 += w * x0 * x2
        g11 += w * x1 * x1
        g12 += w * x1 * x2
        g22 += w * x2 * x2
        yi = ys[i]
        r0 += w * x0 * yi
        r1 += w * x1 * yi
        r2 += w * x2 * yi
    g00 += lam
    g11 += lam
    g22 += lam
    r0 += lam * pa
    r1 += lam * pb
    r2 += lam * pc
    det = g00 * (g11 * g22 - g12 * g12) - g01 * (g01 * g22 - g02 * g12) + g02 * (g01 * g12 - g02 * g11)
    if abs(det) < 1e-18:
        return pa, pb, pc
    inv00 = (g11 * g22 - g12 * g12) / det
    inv01 = (g02 * g12 - g01 * g22) / det
    inv02 = (g01 * g12 - g02 * g11) / det
    inv11 = (g00 * g22 - g02 * g02) / det
    inv12 = (g01 * g02 - g00 * g12) / det
    inv22 = (g00 * g11 - g01 * g01) / det
    return (
        float(inv00 * r0 + inv01 * r1 + inv02 * r2),
        float(inv01 * r0 + inv11 * r1 + inv12 * r2),
        float(inv02 * r0 + inv12 * r1 + inv22 * r2),
    )


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def clip_by_vega(vg: float, core: bool) -> int:
    if core:
        if vg >= 0.35:
            return 20
        if vg >= 0.14:
            return 14
        return 8
    if vg >= 0.25:
        return 10
    return 5


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        store = _parse_td(getattr(state, "traderData", None))

        csv_day = int(getattr(state, "backtest_csv_day", store.get("csv_day", 0)))
        store["csv_day"] = csv_day

        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < _WARMUP_STEPS:
            return result, conversions, json.dumps(store, separators=(",", ":"))

        depths = getattr(state, "order_depths", None) or {}
        pos = getattr(state, "position", None) or {}

        bufs_m = store.get("buf_m")
        bufs_iv = store.get("buf_iv")
        bufs_w = store.get("buf_w")
        if not isinstance(bufs_m, list):
            bufs_m, bufs_iv, bufs_w = [], [], []
        a = float(store.get("a", _PA))
        b = float(store.get("b", _PB))
        c = float(store.get("c", _PC))
        step = int(store.get("step", 0))

        depth_u = depths.get(U)
        bb_u, ba_u = best_bid_ask(depth_u)
        if bb_u is None or ba_u is None:
            store.update({"buf_m": bufs_m, "buf_iv": bufs_iv, "buf_w": bufs_w, "a": a, "b": b, "c": c, "step": step})
            return result, conversions, json.dumps(store, separators=(",", ":"))

        S = 0.5 * float(bb_u) + 0.5 * float(ba_u)
        if S <= 0:
            store.update({"buf_m": bufs_m, "buf_iv": bufs_iv, "buf_w": bufs_w, "a": a, "b": b, "c": c, "step": step})
            return result, conversions, json.dumps(store, separators=(",", ":"))

        T = t_years(csv_day, ts)
        sqrtT = math.sqrt(T)

        if (ts // 100) % 4 == 0:
            for sym in VOUCHERS:
                d = depths.get(sym)
                bbp, bap = best_bid_ask(d)
                if bbp is None or bap is None:
                    continue
                mid = 0.5 * float(bbp) + 0.5 * float(bap)
                K = float(sym.split("_")[1])
                iv = implied_vol_newton(mid, S, K, T, 0.0)
                if iv is None:
                    continue
                m = math.log(K / S) / sqrtT
                bufs_m.append(m)
                bufs_iv.append(iv)
                bufs_w.append(max(vega(S, K, T, iv, 0.0), 1e-6))
            over = len(bufs_m) - _BUFFER_CAP
            if over > 0:
                bufs_m = bufs_m[over:]
                bufs_iv = bufs_iv[over:]
                bufs_w = bufs_w[over:]

        step += 1
        if step % _RECENTER_EVERY == 0 and len(bufs_m) >= 30:
            a, b, c = ridge_abc(bufs_m, bufs_iv, bufs_w, _RIDGE_LAMBDA, _PA, _PB, _PC)

        use_prior_only = step < _PRIOR_ONLY_STEPS
        fa, fb, fc = (_PA, _PB, _PC) if use_prior_only else (a, b, c)

        for sym in VOUCHERS:
            d = depths.get(sym)
            bbp, bap = best_bid_ask(d)
            if bbp is None or bap is None:
                continue
            spr = int(bap - bbp)
            if spr < _MIN_VEV_SPREAD:
                continue
            mid = 0.5 * float(bbp) + 0.5 * float(bap)
            K = float(sym.split("_")[1])
            m = math.log(K / S) / sqrtT
            sig = sigma_from_abc(m, fa, fb, fc)
            fair = bs_call(S, K, T, sig, 0.0)
            vg = vega(S, K, T, sig, 0.0)
            if vg < _MIN_VEGA:
                continue
            th_abs = abs(call_theta(S, K, T, sig, 0.0))
            th_m = theta_edge_multiplier(th_abs) * spread_gate_multiplier(spr)
            edge = mid - fair
            e_ov = _EDGE_OVER_VEGA_MIN * th_m
            if abs(edge) / max(vg, 1e-6) < e_ov:
                continue
            thr = _EDGE_FRAC * float(spr) * th_m
            if abs(edge) < thr:
                continue

            core = sym in CORE_STRIKES
            lim = LIMITS.get(sym, 300)
            p0 = int(pos.get(sym, 0))
            cap = clip_by_vega(vg, core)
            if not core:
                cap = max(1, cap // 2)

            orders_sym: list[Order] = []
            if edge < -thr and p0 < lim:
                q = min(cap, lim - p0)
                if q > 0:
                    orders_sym.append(Order(sym, int(bap), int(q)))
            elif edge > thr and p0 > -lim:
                q = min(cap, lim + p0)
                if q > 0:
                    orders_sym.append(Order(sym, int(bbp), -int(q)))
            if orders_sym:
                result[sym] = orders_sym

        dg = depths.get(GEL)
        gb, ga = best_bid_ask(dg)
        if gb is not None and ga is not None and ga > gb + 1:
            glim = LIMITS.get(GEL, 200)
            gp = int(pos.get(GEL, 0))
            bid_p = gb + 1
            ask_p = ga - 1
            if bid_p < ask_p:
                oo: list[Order] = []
                if gp < glim:
                    oo.append(Order(GEL, bid_p, min(_GEL_SIZE, glim - gp)))
                if gp > -glim:
                    oo.append(Order(GEL, ask_p, -min(_GEL_SIZE, glim + gp)))
                if oo:
                    result[GEL] = oo

        if ba_u > bb_u + 1:
            ulim = LIMITS.get(U, 200)
            up = int(pos.get(U, 0))
            ub = bb_u + 1
            ua = ba_u - 1
            if ub < ua:
                uo: list[Order] = []
                if up < ulim:
                    uo.append(Order(U, ub, min(_EXTRACT_MM, ulim - up)))
                if up > -ulim:
                    uo.append(Order(U, ua, -min(_EXTRACT_MM, ulim + up)))
                if uo:
                    result[U] = uo

        store.update(
            {
                "buf_m": bufs_m,
                "buf_iv": bufs_iv,
                "buf_w": bufs_w,
                "a": a,
                "b": b,
                "c": c,
                "step": step,
                "csv_day": csv_day,
            }
        )
        return result, conversions, json.dumps(store, separators=(",", ":"))
