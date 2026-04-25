#!/usr/bin/env python3
"""Offline tape analysis: IV smile vs volume-weighted WLS residual (Round 3 days 0–2).

Timing (authoritative): round3work/round3description.txt — CSV historical day 0 open
TTE=8d, day 1 → 7d, day 2 → 6d; intraday winding matches
round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py
(dte_eff = start-of-day DTE minus ~1 session-day; T = dte_eff/365, r=0).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(ts: int) -> float:
    return (int(ts) // 100) / 10_000.0


def dte_effective(day: int, ts: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(ts), 1e-6)


def t_years(day: int, ts: int) -> float:
    return dte_effective(day, ts) / 365.0


def bs_call(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def implied_vol_call(mid: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-9 or mid >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    lo, hi = 1e-4, 10.0
    for _ in range(64):
        m = 0.5 * (lo + hi)
        if bs_call(S, K, T, m, r) > mid:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def vega_call(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0 or sig <= 1e-12:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    return float(S * norm.pdf(d1) * math.sqrt(T))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STEP = 100
    resids: list[float] = []
    weights: list[float] = []
    spreads: list[float] = []
    vegas: list[float] = []

    for day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        bv = df.pivot_table(index="timestamp", columns="product", values="bid_volume_1", aggfunc="first")
        av = df.pivot_table(index="timestamp", columns="product", values="ask_volume_1", aggfunc="first")
        bp = df.pivot_table(index="timestamp", columns="product", values="bid_price_1", aggfunc="first")
        ap = df.pivot_table(index="timestamp", columns="product", values="ask_price_1", aggfunc="first")

        for i, ts in enumerate(pvt.index):
            if i % STEP != 0:
                continue
            S = float(pvt.loc[ts, "VELVETFRUIT_EXTRACT"])
            if not np.isfinite(S) or S <= 0:
                continue
            T = t_years(day, int(ts))
            sqrtT = math.sqrt(T)
            xs, ys, ws = [], [], []
            for v in VOU:
                if v not in pvt.columns:
                    continue
                mid = float(pvt.loc[ts, v])
                if not np.isfinite(mid):
                    continue
                K = float(v.split("_")[1])
                iv = implied_vol_call(mid, S, K, T)
                if not np.isfinite(iv):
                    continue
                m_t = math.log(K / S) / sqrtT
                bvv = float(bv.loc[ts, v]) if v in bv.columns and np.isfinite(bv.loc[ts, v]) else 0.0
                avv = float(av.loc[ts, v]) if v in av.columns and np.isfinite(av.loc[ts, v]) else 0.0
                w = max(bvv + avv + 1.0, 1.0)
                xs.append(m_t)
                ys.append(iv)
                ws.append(w)
            if len(xs) < 5:
                continue
            X = np.c_[np.array(xs) ** 2, xs, np.ones(len(xs))]
            wsqrt = np.sqrt(np.array(ws))
            Xw = X * wsqrt[:, None]
            yw = np.array(ys) * wsqrt
            coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
            for v in VOU:
                if v not in pvt.columns:
                    continue
                mid = float(pvt.loc[ts, v])
                if not np.isfinite(mid):
                    continue
                K = float(v.split("_")[1])
                iv = implied_vol_call(mid, S, K, T)
                if not np.isfinite(iv):
                    continue
                m_t = math.log(K / S) / sqrtT
                bvv = float(bv.loc[ts, v]) if v in bv.columns and np.isfinite(bv.loc[ts, v]) else 0.0
                avv = float(av.loc[ts, v]) if v in av.columns and np.isfinite(av.loc[ts, v]) else 0.0
                w = max(bvv + avv + 1.0, 1.0)
                pred_iv = float(coef[0] * m_t * m_t + coef[1] * m_t + coef[2])
                resid = iv - pred_iv
                resids.append(resid)
                weights.append(w)
                vegas.append(vega_call(S, K, T, iv))
                if v in bp.columns and v in ap.columns:
                    bpx = bp.loc[ts, v]
                    apx = ap.loc[ts, v]
                    if np.isfinite(bpx) and np.isfinite(apx):
                        spreads.append(float(apx - bpx))

    r = np.asarray(resids, dtype=float)
    w = np.asarray(weights, dtype=float)
    summary = {
        "n_observations": int(len(r)),
        "mean_residual_iv": float(np.mean(r)),
        "std_residual_iv": float(np.std(r)),
        "corr_abs_residual_log_weight": float(np.corrcoef(np.abs(r), np.log(w))[0, 1]),
        "mean_spread_top_of_book": float(np.mean(spreads)) if spreads else None,
        "mean_vega_at_fit_iv": float(np.mean(vegas)) if vegas else None,
        "timing": {
            "dte_at_csv_day_open": {"0": 8, "1": 7, "2": 6},
            "intraday_wind": "dte_eff = dte_open - (timestamp//100)/10000",
            "T_years": "dte_eff / 365",
            "r": 0.0,
        },
        "method": {
            "iv": "Black–Scholes European call, bisection on sigma",
            "smile": "WLS quadratic in m_t=log(K/S)/sqrt(T); weights = 1 + bid_vol_1 + ask_vol_1",
            "residual": "IV - fitted_smile(m_t)",
        },
    }
    out_path = OUT / "iv_residual_volume_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
