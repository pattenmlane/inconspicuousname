"""Offline grid: how EWMA alpha affects smoothness of vega-weighted quadratic IV fit on day 0 subsample."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "ewma_alpha_offline_grid.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def dte_effective(day: int, timestamp: int) -> float:
    d0 = 8 - int(day)
    prog = (int(timestamp) // 100) / 10_000.0
    return max(float(d0) - prog, 1e-6)


def t_years(day: int, ts: int) -> float:
    return dte_effective(day, ts) / 365.0


def bs_call(S, K, T, sig, r=0.0):
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, sig, r=0.0):
    if T <= 0 or sig <= 1e-12:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    return S * norm.pdf(d1) * math.sqrt(T)


def iv_call(mid, S, K, T, r=0.0):
    if mid <= max(S - K, 0) + 1e-6 or mid >= S - 1e-6 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig):
        return bs_call(S, K, T, sig, r) - mid

    try:
        if f(1e-5) > 0 or f(12.0) < 0:
            return float("nan")
        return brentq(f, 1e-5, 12.0)
    except ValueError:
        return float("nan")


def main() -> None:
    day = 0
    step = 200
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    ts_list = sorted(df["timestamp"].unique())[::step]
    series_raw: list[np.ndarray] = []
    for ts in ts_list:
        sub = df[df["timestamp"] == ts]
        ex = sub[sub["product"] == "VELVETFRUIT_EXTRACT"]
        if ex.empty:
            continue
        S = float(ex.iloc[0]["mid_price"])
        T = t_years(day, int(ts))
        sqrtT = math.sqrt(T)
        xs, ys, w = [], [], []
        for v in VOUCHERS:
            r0 = sub[sub["product"] == v]
            if r0.empty:
                continue
            mid = float(r0.iloc[0]["mid_price"])
            K = float(v.split("_")[1])
            iv = iv_call(mid, S, K, T)
            if not np.isfinite(iv):
                continue
            m_t = math.log(K / S) / sqrtT
            xs.append(m_t)
            ys.append(iv)
            w.append(bs_vega(S, K, T, iv) + 1e-6)
        if len(xs) < 6:
            continue
        coeff = np.polyfit(np.asarray(xs), np.asarray(ys), 2, w=np.asarray(w))
        series_raw.append(coeff)

    if len(series_raw) < 5:
        print("too few", file=sys.stderr)
        sys.exit(1)

    raw = np.stack(series_raw, axis=0)
    alphas = [0.02, 0.03, 0.04, 0.06]
    results = []
    for a in alphas:
        ema = raw[0].copy()
        path = [ema.copy()]
        for t in range(1, len(raw)):
            ema = (1 - a) * ema + a * raw[t]
            path.append(ema.copy())
        path_a = np.stack(path, axis=0)
        # smoothness: mean L2 step size of coeff vector
        steps = np.linalg.norm(np.diff(path_a, axis=0), axis=1)
        results.append(
            {
                "ewma_alpha": a,
                "mean_coeff_step_l2": float(np.mean(steps)),
                "final_coeff": [float(x) for x in path_a[-1]],
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "note": "Offline only (day 0, coarse timestamps). Lower alpha = smoother coeff path (smaller steps).",
                "alphas": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
