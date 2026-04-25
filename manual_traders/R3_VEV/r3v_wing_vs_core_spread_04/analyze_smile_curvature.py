#!/usr/bin/env python3
"""
Round-3 tape: IV smile curvature vs quadratic core (m_t = log(K/S)/sqrt(T)).
TTE/DTE mapping matches round3work/round3description.txt + intraday wind-down
used in round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py:
  CSV day column 0,1,2 -> calendar DTE at open 8,7,6; session winds ~1 DTE over the day.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years_effective(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def implied_vol_call(market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-9:
        return float("nan")
    if market >= S - 1e-9:
        return float("nan")
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    lo, hi = 1e-5, 15.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-8, rtol=1e-8)
    except ValueError:
        return float("nan")


def bs_call_greeks(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> dict:
    if T <= 1e-12 or sigma <= 1e-12 or S <= 0 or K <= 0:
        return {"delta": float("nan"), "gamma": float("nan"), "vega": float("nan"), "theta": float("nan")}
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * v)
    vega = S * norm.pdf(d1) * math.sqrt(T)
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * norm.cdf(d2)
    )
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta)}


def load_wide(day: int) -> pd.DataFrame:
    path = DATA / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
    if "VELVETFRUIT_EXTRACT" not in pvt.columns:
        raise RuntimeError("missing underlying")
    vcols = [v for v in VOUCHERS if v in pvt.columns]
    out = pvt[["VELVETFRUIT_EXTRACT"] + vcols].copy()
    out.columns = ["S"] + vcols
    return out


def spread_width(df: pd.DataFrame, product: str, sample_ts: list[int]) -> float:
    sub = df[(df["product"] == product) & (df["timestamp"].isin(sample_ts))]
    if sub.empty:
        return float("nan")
    widths = []
    for _, r in sub.iterrows():
        bp = [r[f"bid_price_{i}"] for i in (1, 2, 3) if pd.notna(r.get(f"bid_price_{i}"))]
        ap = [r[f"ask_price_{i}"] for i in (1, 2, 3) if pd.notna(r.get(f"ask_price_{i}"))]
        if bp and ap:
            widths.append(float(min(ap) - max(bp)))
    return float(np.nanmean(widths)) if widths else float("nan")


def main() -> None:
    step = 200
    wing_ks = (4000, 4500, 5400, 5500, 6000, 6500)
    core_ks = (5100, 5200, 5300)
    resid_spreads: list[float] = []
    wing_minus_core: list[float] = []
    coeffs_a: list[float] = []

    for day in (0, 1, 2):
        raw = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        wide = load_wide(day)
        idx = wide.index[::step]
        sample_ts = [int(x) for x in idx[:50]]

        for ts in idx:
            ts_i = int(ts)
            row = wide.loc[ts]
            S = float(row["S"])
            if S <= 0:
                continue
            T = t_years_effective(day, ts_i)
            if T <= 0:
                continue
            sqrtT = math.sqrt(T)
            xs: list[float] = []
            ys: list[float] = []
            iv_by_k: dict[int, float] = {}
            for v in VOUCHERS:
                if v not in row.index:
                    continue
                mid = float(row[v])
                K = int(v.split("_")[1])
                iv = implied_vol_call(mid, S, K, T, 0.0)
                if not np.isfinite(iv):
                    continue
                m_t = math.log(K / S) / sqrtT
                xs.append(m_t)
                ys.append(iv)
                iv_by_k[K] = iv
            if len(xs) < 6:
                continue
            coeff = np.polyfit(np.asarray(xs), np.asarray(ys), 2)
            a, b, c = float(coeff[0]), float(coeff[1]), float(coeff[2])
            coeffs_a.append(a)

            def pred(m: float) -> float:
                return a * m * m + b * m + c

            wing_ivs = [iv_by_k[k] for k in wing_ks if k in iv_by_k]
            core_ivs = [iv_by_k[k] for k in core_ks if k in iv_by_k]
            if len(wing_ivs) < 3 or len(core_ivs) < 2:
                continue
            w_mean = float(np.mean(wing_ivs))
            c_mean = float(np.mean(core_ivs))
            wing_minus_core.append(w_mean - c_mean)

            resids = []
            for v in VOUCHERS:
                if v not in row.index:
                    continue
                K = int(v.split("_")[1])
                if K not in iv_by_k:
                    continue
                m_t = math.log(K / S) / sqrtT
                resids.append(iv_by_k[K] - pred(m_t))
            if resids:
                resid_spreads.append(float(np.max(resids) - np.min(resids)))

        for pr in ("VEV_5200", "VELVETFRUIT_EXTRACT"):
            w = spread_width(raw, pr, sample_ts)
            print(f"day{day} mean spread width {pr} (subsample): {w:.4f}")

    summary = {
        "dte_mapping": (
            "CSV `day` 0,1,2 per round3description example maps to TTE 8d,7d,6d at session open; "
            "intraday DTE decreases ~1 day over timestamps 0..999900 (step 100) per plot_iv_smile_round3."
        ),
        "subsample_step": step,
        "n_fits": len(coeffs_a),
        "quadratic_a_iv_vs_m_t": {
            "mean": float(np.mean(coeffs_a)) if coeffs_a else None,
            "std": float(np.std(coeffs_a)) if coeffs_a else None,
        },
        "mean_wing_iv_minus_core_iv": float(np.mean(wing_minus_core)) if wing_minus_core else None,
        "std_wing_iv_minus_core_iv": float(np.std(wing_minus_core)) if wing_minus_core else None,
        "mean_range_quadratic_residual_across_strikes": float(np.mean(resid_spreads)) if resid_spreads else None,
        "wing_strikes": list(wing_ks),
        "core_strikes": list(core_ks),
        "bs_assumption": "European call, r=0, IV from mid via Brent on BS price; greeks at fitted IV.",
    }

    out_path = OUT_DIR / "smile_curvature_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
