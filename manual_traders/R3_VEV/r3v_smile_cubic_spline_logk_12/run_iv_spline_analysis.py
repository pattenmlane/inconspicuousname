#!/usr/bin/env python3
"""
Offline analysis: Black–Scholes implied vol from mids vs log(K) cubic spline fair.
TTE from round3work/round3description.txt: historical tape day index d (0,1,2 in
prices_round_3_day_d.csv) maps to TTE = (8 - d) days (same pattern as doc example:
historical day 1 -> 8d, day 2 -> 7d, day 3 -> 6d; 0-based file day_0 = first day -> 8d).
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
VEV = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
STRIKES = [int(s.split("_")[1]) for s in VEV]
UNDER = "VELVETFRUIT_EXTRACT"
R = 0.0


def _read_day(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            rows.append(row)
    return rows


def _mid(row: dict) -> float | None:
    m = row.get("mid_price")
    if m is None or m == "":
        return None
    try:
        return float(m)
    except ValueError:
        return None


def _spread(row: dict) -> float | None:
    bp = row.get("bid_price_1")
    ap = row.get("ask_price_1")
    if not bp or not ap:
        return None
    try:
        b, a = float(bp), float(ap)
        if a <= b:
            return None
        return a - b
    except ValueError:
        return None


def bs_price_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)


def implied_vol(S: float, K: float, T: float, price: float, lo=1e-6, hi=5.0) -> float | None:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if price < intrinsic - 1e-9:
        return None
    hi_c = bs_price_call(S, K, T, hi)
    if price > hi_c:
        return None
    lo_c = bs_price_call(S, K, T, lo)
    if price < lo_c:
        lo = 1e-8
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def tte_years(day_idx: int) -> float:
    """Tape file day index 0,1,2 -> TTE 8,7,6 calendar days per round3description."""
    dte = 8 - int(day_idx)
    return max(dte, 1) / 365.25


def load_snapshot(rows: Iterable[dict], ts: int, day_idx: int) -> tuple[dict[str, float], float | None]:
    """Return mid prices for all products at timestamp, and underlying mid."""
    mids: dict[str, float] = {}
    for row in rows:
        if int(row["timestamp"]) != ts:
            continue
        p = row["product"]
        m = _mid(row)
        if m is not None:
            mids[p] = m
    return mids, mids.get(UNDER)


def analyze_day(day_idx: int, sample_stride: int = 500) -> dict:
    path = DATA / f"prices_round_3_day_{day_idx}.csv"
    rows = _read_day(path)
    timestamps = sorted({int(r["timestamp"]) for r in rows})
    T = tte_years(day_idx)
    iv_residuals_insample: list[float] = []
    iv_residuals_loo: list[float] = []
    spreads: list[float] = []
    dS_list: list[float] = []
    div_atm_list: list[float] = []

    prev_S: float | None = None
    prev_iv_atm: float | None = None

    for ts in timestamps[::sample_stride]:
        mids, S = load_snapshot(rows, ts, day_idx)
        if S is None or S <= 0:
            continue
        logKs: list[float] = []
        ivs: list[float] = []
        for sym, K in zip(VEV, STRIKES):
            mid = mids.get(sym)
            if mid is None:
                continue
            iv = implied_vol(S, float(K), T, mid)
            if iv is None:
                continue
            logKs.append(math.log(K))
            ivs.append(iv)
            for row in rows:
                if int(row["timestamp"]) == ts and row["product"] == sym:
                    sp = _spread(row)
                    if sp is not None:
                        spreads.append(sp)
                    break
        if len(ivs) < 5:
            continue
        order = np.argsort(logKs)
        x = np.array([logKs[i] for i in order], dtype=float)
        y = np.array([ivs[i] for i in order], dtype=float)
        # Robust: drop point farthest from median IV if range is huge (one bad tick)
        if float(np.max(y) - np.min(y)) > 0.5 and len(y) >= 6:
            med = float(np.median(y))
            dev = np.abs(y - med)
            drop = int(np.argmax(dev))
            x = np.delete(x, drop)
            y = np.delete(y, drop)
        if len(x) < 4:
            continue
        cs = CubicSpline(x, y, bc_type="natural")
        for xi, yi in zip(x, y):
            iv_residuals_insample.append(float(yi - cs(xi)))
        # Leave-one-out cross-validation on logK nodes (robustness / interpolation error)
        if len(x) >= 5:
            for k in range(len(x)):
                mask = np.ones(len(x), dtype=bool)
                mask[k] = False
                x_loo = x[mask]
                y_loo = y[mask]
                try:
                    cs_loo = CubicSpline(x_loo, y_loo, bc_type="natural")
                    iv_residuals_loo.append(float(y[k] - cs_loo(x[k])))
                except Exception:
                    pass
        # ATM-ish: IV at strike closest to underlying
        j = int(np.argmin(np.abs(np.exp(x) - S)))
        iv_atm = float(y[j])
        if prev_S is not None and prev_iv_atm is not None:
            dS_list.append(S - prev_S)
            div_atm_list.append(iv_atm - prev_iv_atm)
        prev_S = S
        prev_iv_atm = iv_atm

    corr_dS_div = None
    if len(dS_list) >= 5:
        corr_dS_div = float(np.corrcoef(dS_list, div_atm_list)[0, 1])

    return {
        "day_idx": day_idx,
        "T_years": T,
        "TTE_days": 8 - day_idx,
        "n_iv_residual_insample": len(iv_residuals_insample),
        "iv_insample_rmse": float(math.sqrt(statistics.mean(r * r for r in iv_residuals_insample)))
        if iv_residuals_insample
        else None,
        "n_iv_residual_loo": len(iv_residuals_loo),
        "iv_loo_rmse": float(math.sqrt(statistics.mean(r * r for r in iv_residuals_loo))) if iv_residuals_loo else None,
        "spread_mean": float(statistics.mean(spreads)) if spreads else None,
        "spread_median": float(statistics.median(spreads)) if spreads else None,
        "corr_dS_dIV_atm": corr_dS_div,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for d in (0, 1, 2):
        summaries.append(analyze_day(d))
    out_csv = OUT / "iv_spline_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    # Small parameter grid: edge vs spline RMSE sensitivity (re-simulate fewer samples)
    grid_rows = []
    for stride in (200, 500, 1000):
        rmses = []
        for d in (0, 1, 2):
            s = analyze_day(d, sample_stride=stride)
            rmses.append(s["iv_loo_rmse"] or 0.0)
        grid_rows.append({"sample_stride": stride, "mean_iv_loo_rmse": float(np.mean(rmses))})
    grid_path = OUT / "grid_sample_stride.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_stride", "mean_iv_loo_rmse"])
        w.writeheader()
        w.writerows(grid_rows)

    print("Wrote", out_csv, "and", grid_path)
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
