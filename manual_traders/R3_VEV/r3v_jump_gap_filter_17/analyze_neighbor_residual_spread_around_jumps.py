"""
Tape: ptp(VEV wall_mid - LOO BS fair) by timestamp vs |dS| (extract); jump = |dS| >= 3.

Fast path: one pivot per day, no per-tick df scans.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"))
from plot_iv_smile_round3 import t_years_effective  # noqa: E402

R = 0.0
_SQRT2PI = math.sqrt(2.0 * math.pi)
JUMP = 3.0
ROBUST_IV = 0.35
EX = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_SYMS = [f"VEV_{k}" for k in STRIKES]
PROD = [EX] + VEV_SYMS


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * math.exp(-R * T) * _norm_cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    return S * math.sqrt(T) * _norm_pdf(d1)


def implied_vol(
    S: float, K: float, T: float, price: float, initial: float | None = None
) -> float | None:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if price < intrinsic - 1e-9:
        return None
    if bs_call(S, K, T, 4.5) < price - 1e-9:
        return None
    sigma = 0.28 if initial is None else max(1e-4, min(float(initial), 4.5))
    for _ in range(8):
        th = bs_call(S, K, T, sigma) - price
        if abs(th) < 1e-7:
            return sigma
        vg = bs_vega(S, K, T, sigma)
        if vg < 1e-14:
            break
        sigma -= th / vg
        sigma = max(1e-6, min(sigma, 4.5))
    lo, hi = 1e-5, 4.5
    if bs_call(S, K, T, lo) > price or bs_call(S, K, T, hi) < price:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def robust_spline_nodes(
    logks: np.ndarray, ivs: np.ndarray, robust_iv_range: float
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(logks) < 4:
        return None
    order = np.argsort(logks)
    x = logks[order].astype(float)
    y = ivs[order].astype(float)
    if float(np.max(y) - np.min(y)) > robust_iv_range and len(y) >= 6:
        med = float(np.median(y))
        drop = int(np.argmax(np.abs(y - med)))
        x = np.delete(x, drop)
        y = np.delete(y, drop)
    if len(x) < 4:
        return None
    return x, y


def index_of_logk(x: np.ndarray, logk: float) -> int | None:
    for j in range(len(x)):
        if abs(float(x[j]) - logk) < 1e-5:
            return j
    return None


def loo_spline_iv_at(x: np.ndarray, y: np.ndarray, j: int, logk: float) -> float:
    if len(x) >= 5:
        xl = np.delete(x, j)
        yl = np.delete(y, j)
        if len(xl) >= 4:
            cs = CubicSpline(xl, yl, bc_type="natural")
            return float(cs(float(logk)))
    cs = CubicSpline(x, y, bc_type="natural")
    return float(cs(float(logk)))


def fair_iv_for_strike(x: np.ndarray, y: np.ndarray, logk: float) -> float:
    j = index_of_logk(x, logk)
    if j is None or len(x) < 4 or len(x) < 5:
        cs = CubicSpline(x, y, bc_type="natural")
        return float(cs(float(logk)))
    return loo_spline_iv_at(x, y, j, logk)


def main() -> None:
    ptp_rows: list[dict] = []
    for day in (0, 1, 2):
        path = REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        df = df[df["product"].isin(PROD)]
        pvt = (
            df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="last")
            .sort_index()
        )
        if EX not in pvt.columns:
            continue
        s = pvt[EX].astype(float)
        dS = s.diff().abs().fillna(0.0)
        iv_prev: dict[str, float] = {}
        for t in pvt.index:
            row = pvt.loc[t]
            S = float(row[EX])
            if not np.isfinite(S) or S <= 0:
                continue
            tv = t_years_effective(int(day), int(t))
            if tv <= 0:
                continue
            logks: list[float] = []
            ivs: list[float] = []
            for sym, K in zip(VEV_SYMS, STRIKES):
                if sym not in row.index or pd.isna(row[sym]):
                    continue
                m = float(row[sym])
                p0 = iv_prev.get(sym)
                init = float(p0) if isinstance(p0, (int, float)) else None
                iv = implied_vol(S, float(K), float(tv), m, initial=init)
                if iv is None:
                    continue
                iv_prev[sym] = iv
                logks.append(math.log(float(K)))
                ivs.append(iv)
            if len(logks) < 4:
                continue
            rnodes = robust_spline_nodes(np.array(logks), np.array(ivs), ROBUST_IV)
            if rnodes is None:
                continue
            xa, ya = rnodes
            resids: list[float] = []
            for sym, K in zip(VEV_SYMS, STRIKES):
                if sym not in row.index or pd.isna(row[sym]):
                    continue
                m = float(row[sym])
                lk = math.log(float(K))
                sig = fair_iv_for_strike(xa, ya, lk)
                fv = bs_call(S, float(K), float(tv), float(sig))
                resids.append(m - fv)
            if len(resids) < 2:
                continue
            ds0 = float(dS.loc[t])
            ptp = float(max(resids) - min(resids))
            ptp_rows.append({"day": int(day), "dS": ds0, "ptp": ptp, "jump": int(ds0 >= JUMP)})
    out = pd.DataFrame(ptp_rows)
    j = out.loc[out["jump"] == 1, "ptp"]
    n = out.loc[out["jump"] == 0, "ptp"]
    summ = {
        "jump_threshold_abs_dS": JUMP,
        "n": int(len(out)),
        "n_jump": int((out["jump"] == 1).sum()),
        "ptp_median_all": float(out["ptp"].median()) if len(out) else 0.0,
        "ptp_median_jump": float(j.median()) if len(j) else None,
        "ptp_median_nojump": float(n.median()) if len(n) else None,
        "ptp_p90_jump": float(j.quantile(0.9)) if len(j) else None,
        "ptp_p90_nojump": float(n.quantile(0.9)) if len(n) else None,
    }
    p = Path(__file__).resolve().parent / "analysis_neighbor_residual_ptp_jump.json"
    p.write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
