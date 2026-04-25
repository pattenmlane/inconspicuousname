"""
Round 3 tapes: |mid - BS(poly IV)| vs |mid - BS(spline IV)| by strike, same model shape as v12
(wing reg lambda, anchor nodes 25% outside span).

DTE: round3 work — CSV day d -> 8-d at open, minus intraday progress (round3work/round3description.txt).
Tape format: long (one row per product per timestamp), see analyze_round3_smile.py.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "spline_poly_edge_by_strike.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
WING_REG_LAMBDA = 0.1
ANCHOR_FRAC = 0.25
STEP = 200


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(ts: int) -> float:
    return (int(ts) // 100) / 10_000.0


def t_years_effective(day: int, ts: int) -> float:
    dte = max(float(dte_from_csv_day(day)) - intraday_progress(ts), 1e-6)
    return dte / 365.0


def bs_call(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_vega(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    return S * norm.pdf(d1) * math.sqrt(T)


def iv_from_mid(m: float, S: float, K: float, T: float) -> float | None:
    if T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if m <= intrinsic + 1e-6 or m >= S - 1e-6:
        return None

    def f(sig: float) -> float:
        return bs_call(S, K, T, sig) - m

    lo, hi = 1e-5, 12.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-6, rtol=1e-6))
    except ValueError:
        return None


def one_row(
    S: float,
    mids: dict[str, float],
    T: float,
) -> dict[str, tuple[float, float]] | None:
    sqrtT = math.sqrt(T) if T > 0 else 1e-6
    x_nodes: dict[str, float] = {}
    ivs: dict[str, float] = {}
    for v in VOUCHERS:
        if v not in mids:
            continue
        K = float(v.split("_")[1])
        iv = iv_from_mid(mids[v], S, K, T)
        if iv is None:
            continue
        ivs[v] = iv
        x_nodes[v] = math.log(K / S) / sqrtT
    if len(ivs) < 6:
        return None
    x_list, y_list, w_list = [], [], []
    for v, iv in ivs.items():
        K = float(v.split("_")[1])
        vg = bs_vega(S, K, T, max(iv, 1e-4))
        x_list.append(x_nodes[v])
        y_list.append(iv)
        w_list.append(max(vg, 1e-6))
    x_arr = np.asarray(x_list, dtype=float)
    y_arr = np.asarray(y_list, dtype=float)
    w_arr = np.asarray(w_list, dtype=float)
    coef2 = np.polyfit(x_arr, y_arr, 2, w=w_arr)
    y_base = np.polyval(coef2, x_arr)
    y_reg = (1.0 - WING_REG_LAMBDA) * y_arr + WING_REG_LAMBDA * y_base
    ord_idx = np.argsort(x_arr)
    x_sort, y_sort = x_arr[ord_idx], y_reg[ord_idx]
    span = float(x_sort[-1] - x_sort[0]) if len(x_sort) > 1 else 0.0
    if span <= 0:
        return None
    xl = x_sort[0] - ANCHOR_FRAC * span
    xr = x_sort[-1] + ANCHOR_FRAC * span
    yl, yr = float(np.polyval(coef2, xl)), float(np.polyval(coef2, xr))
    x_fit = np.concatenate([[xl], x_sort, [xr]])
    y_fit = np.concatenate([[yl], y_sort, [yr]])
    spl = CubicSpline(x_fit, y_fit, bc_type="natural")
    c2, c1, c0 = float(coef2[0]), float(coef2[1]), float(coef2[2])
    out: dict[str, tuple[float, float]] = {}
    for v in VOUCHERS:
        if v not in ivs:
            continue
        K = int(v.split("_")[1])
        x = float(math.log(K / S) / sqrtT)
        ivp = max(1e-4, min(8.0, float(c2 * x * x + c1 * x + c0)))
        ivspline = max(1e-4, min(8.0, float(spl(x))))
        fp = bs_call(S, float(K), T, ivp)
        fs = bs_call(S, float(K), T, ivspline)
        mid = mids[v]
        out[v] = (abs(mid - fp), abs(mid - fs))
    return out


def main() -> None:
    abs_poly = defaultdict(list)
    abs_spl = defaultdict(list)
    for csv_day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{csv_day}.csv"
        df = pd.read_csv(path, sep=";")
        ts_list = sorted(df["timestamp"].unique())[::STEP]
        for ts in ts_list:
            sub = df[df["timestamp"] == ts]
            ex = sub[sub["product"] == U]
            if ex.empty:
                continue
            S = float(ex.iloc[0]["mid_price"])
            if not math.isfinite(S) or S <= 0:
                continue
            T = t_years_effective(csv_day, int(ts))
            mids: dict[str, float] = {}
            for v in VOUCHERS:
                r0 = sub[sub["product"] == v]
                if r0.empty:
                    continue
                mids[v] = float(r0.iloc[0]["mid_price"])
            o = one_row(S, mids, T)
            if o is None:
                continue
            for v, (ap, ase) in o.items():
                abs_poly[v].append(ap)
                abs_spl[v].append(ase)
    if not any(abs_poly.values()):
        print("no samples", file=sys.stderr)
        sys.exit(1)
    by_v: dict[str, dict[str, float]] = {}
    for v in VOUCHERS:
        if not abs_poly[v]:
            continue
        pa = np.asarray(abs_poly[v], float)
        sa = np.asarray(abs_spl[v], float)
        med_p, med_s = float(np.median(pa)), float(np.median(sa))
        by_v[v] = {
            "n": float(len(pa)),
            "median_abs_poly": med_p,
            "median_abs_spline": med_s,
            "ratio_poly_over_spline": float(med_p / max(med_s, 1e-9)),
        }
    payload = {
        "step": STEP,
        "wing_reg_lambda": WING_REG_LAMBDA,
        "by_voucher": by_v,
        "note": "Instantaneous cross-section (no sticky EWMA). Ratios show where poly would widen fair vs spline.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
