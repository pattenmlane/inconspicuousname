"""
Offline: sequential EWMA on smile coeffs from tape; measure fair drift and BS theta at VEV_5200.

For each EMA alpha in a small grid, walk timestamps (day 0-2, step 25), recompute snap smile
(vega-weighted polyfit), EMA the coeffs, then BS fair and analytical call theta for K=5200.
Reports median |Δfair| per step and median |theta| (time decay per year, scaled to ~per-tick
via ΔT = 1 session-day / 10k steps in same dte_wind units as dte_effective / 365).

DTE: 8 - csv_day at open, intraday wind per plot_iv_smile.
Output: analysis_outputs/smile_ewma_fair_drift_5200.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "smile_ewma_fair_drift_5200.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
V = [f"VEV_{k}" for k in STRIKES]
K0 = 5200.0
FOCAL = f"VEV_{int(K0)}"


def dte_e(day: int, ts: int) -> float:
    return max(8.0 - float(day) - (int(ts) // 100) / 10000.0, 1e-6)


def t_y(day: int, ts: int) -> float:
    return dte_e(day, ts) / 365.0


def bsc(S: float, K: float, T: float, s: float) -> float:
    if T <= 0 or s <= 1e-12:
        return max(S - K, 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def implied_vol(market: float, S: float, K: float, T: float) -> float | None:
    if market <= max(S - K, 0) + 1e-6 or market >= S - 1e-6 or S <= 0 or K <= 0 or T <= 0:
        return None

    def f(sig: float) -> float:
        return bsc(S, K, T, sig) - market

    try:
        if f(1e-5) > 0 or f(12) < 0:
            return None
        return float(brentq(f, 1e-5, 12, xtol=1e-7, rtol=1e-7))
    except ValueError:
        return None


def bs_call_theta(S: float, K: float, T: float, sigma: float) -> float:
    """dPrice/dT (r=0) Black–Scholes call theta, per year of calendar T."""
    if T <= 0 or sigma <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
    return -(S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))


def one_alpha(alpha: float, step: int) -> dict:
    ema: list[float] | None = None
    thetas: list[float] = []
    fair_hist: list[float] = []

    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        for ts in sorted(df["timestamp"].unique())[::step]:
            sub = df.loc[df["timestamp"] == ts]
            ex = sub.loc[sub["product"] == "VELVETFRUIT_EXTRACT"]
            if ex.empty:
                continue
            S = float(ex.iloc[0]["mid_price"])
            Tq = t_y(day, int(ts))
            st = math.sqrt(Tq)
            xs, ys, ws = [], [], []
            for vv in V:
                r = sub.loc[sub["product"] == vv]
                if r.empty:
                    continue
                row = r.iloc[0]
                if pd.isna(row["bid_price_1"]) or pd.isna(row["ask_price_1"]):
                    continue
                mid = 0.5 * (float(row["bid_price_1"]) + float(row["ask_price_1"]))
                K = float(vv.split("_")[1])
                i = implied_vol(mid, S, K, Tq)
                if i is None:
                    continue
                mt = math.log(K / S) / st
                d1 = (math.log(S / K) + 0.5 * i * i * Tq) / (i * math.sqrt(Tq))
                vg = S * norm.pdf(d1) * math.sqrt(Tq)
                xs.append(mt)
                ys.append(i)
                ws.append(max(vg, 1e-6))
            if len(xs) < 6:
                continue
            c = list(np.polyfit(np.asarray(xs), np.asarray(ys), 2, w=np.asarray(ws)))
            if ema is None:
                ema = [float(c[0]), float(c[1]), float(c[2])]
            else:
                a = alpha
                ema = [(1 - a) * ema[i] + a * float(c[i]) for i in range(3)]
            m_t0 = math.log(K0 / S) / st
            iv0 = max(1e-4, min(8.0, float(np.polyval(ema, m_t0))))
            fair = bsc(S, K0, Tq, iv0)
            fair_hist.append(fair)
            thetas.append(abs(bs_call_theta(S, K0, Tq, iv0)))
    if len(fair_hist) < 3:
        return {"alpha": alpha, "error": "too_few_points"}
    dfair = [abs(fair_hist[i] - fair_hist[i - 1]) for i in range(1, len(fair_hist))]
    return {
        "alpha": alpha,
        "n_steps": len(fair_hist),
        "median_abs_delta_fair": float(np.median(dfair)),
        "median_abs_theta_per_year": float(np.median(thetas)) if thetas else None,
    }


def main() -> None:
    step = 25
    rows = [one_alpha(a, step) for a in (0.02, 0.03, 0.045, 0.06)]
    rows = [r for r in rows if "median_abs_delta_fair" in r]
    rows.sort(key=lambda r: r["median_abs_delta_fair"])
    payload = {
        "focal_strike": int(K0),
        "focal_voucher": FOCAL,
        "subsample_step": step,
        "dte": "8-csv_day at open, intraday -1d per session",
        "grid": rows,
        "lowest_median_drift_alpha": rows[0]["alpha"] if rows else None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
