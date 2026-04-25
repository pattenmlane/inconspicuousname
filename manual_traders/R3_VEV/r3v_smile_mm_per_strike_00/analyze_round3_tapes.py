#!/usr/bin/env python3
"""
Offline analysis: Round 3 tapes → IV smile / skew, spreads, co-movement with extract.
Timing: round3work/round3description.txt + round3work/plotting/.../plot_iv_smile_round3.py
  CSV day 0→DTE 8 at open, 1→7, 2→6; intraday DTE winds ~1 day per session; T = dte_eff/365, r=0.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"

VEVS = [
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
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(ts: int) -> float:
    return (int(ts) // 100) / 10_000.0


def dte_effective(day: int, ts: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(ts), 1e-6)


def t_years(day: int, ts: int) -> float:
    return dte_effective(day, ts) / 365.0


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def iv_from_mid(mid: float, S: float, K: float, T: float) -> float | None:
    if mid <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-6:
        return None
    hi = 3.0
    lo = 1e-4
    try:
        if bs_call(S, K, T, hi) < mid:
            return None

        def f(sig: float) -> float:
            return bs_call(S, K, T, sig) - mid

        return float(brentq(f, lo, hi, maxiter=80))
    except ValueError:
        return None


def strike_from_name(p: str) -> float:
    return float(p.split("_", 1)[1])


def load_day(day: int) -> pd.DataFrame:
    path = DATA / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    return df


def sample_rows(df: pd.DataFrame, step: int) -> pd.DataFrame:
    return df.iloc[::step]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "timing_note": (
            "DTE at open: csv_day 0→8d, 1→7d, 2→6d (round3description example aligns "
            "simulation round index with historical day). Intraday: subtract ~1 day "
            "linearly over timestamps (Frankfurt-style winding from plot_iv_smile_round3)."
        ),
        "days": {},
    }

    for day in (0, 1, 2):
        df = load_day(day)
        # subsample timestamps for speed
        ts_list = sorted(df["timestamp"].unique())[::200]
        iv_rows = []
        spreads = {p: [] for p in VEVS + [U, H]}

        for ts in ts_list:
            sub = df[df["timestamp"] == ts]
            row_by_p = {r["product"]: r for _, r in sub.iterrows()}
            if U not in row_by_p:
                continue
            Su = row_by_p[U]["mid_price"]
            if Su is None or (isinstance(Su, float) and math.isnan(Su)):
                continue
            Su = float(Su)
            T = t_years(day, int(ts))
            for p in VEVS:
                if p not in row_by_p:
                    continue
                r = row_by_p[p]
                mid = r["mid_price"]
                if mid is None or (isinstance(mid, float) and math.isnan(mid)):
                    continue
                mid = float(mid)
                ap1 = r.get("ask_price_1")
                bp1 = r.get("bid_price_1")
                if ap1 is not None and bp1 is not None and not math.isnan(ap1) and not math.isnan(bp1):
                    spreads[p].append(float(ap1) - float(bp1))
                K = strike_from_name(p)
                iv = iv_from_mid(mid, Su, K, T)
                if iv is None:
                    continue
                km = math.log(K / Su)
                iv_rows.append({"product": p, "km": km, "iv": iv, "mid": mid, "timestamp": int(ts)})

        for p in [U, H]:
            for ts in ts_list:
                sub = df[(df["timestamp"] == ts) & (df["product"] == p)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                ap1 = r.get("ask_price_1")
                bp1 = r.get("bid_price_1")
                if ap1 is not None and bp1 is not None and not math.isnan(ap1) and not math.isnan(bp1):
                    spreads[p].append(float(ap1) - float(bp1))

        # Per-timestamp smile curvature: fit iv = a + b*km + c*km^2 on available strikes
        residuals = []
        atm_iv_spread = []
        for ts in ts_list:
            pts = [x for x in iv_rows if x["timestamp"] == ts]
            if len(pts) < 5:
                continue
            kms = np.array([x["km"] for x in pts])
            ivs = np.array([x["iv"] for x in pts])
            try:
                a, b, c = np.polyfit(kms, ivs, 2)
            except (np.linalg.LinAlgError, ValueError):
                continue
            fit = a * kms**2 + b * kms + c
            residuals.extend((ivs - fit).tolist())
            # ATM-ish: km nearest 0
            j = int(np.argmin(np.abs(kms)))
            atm_iv_spread.append(float(ivs[j]))

        day_summary = {
            "n_iv_points": len(iv_rows),
            "mean_spread_by_product": {
                k: (float(statistics.mean(v)) if v else None) for k, v in spreads.items()
            },
            "iv_smile_poly_rmse": float(math.sqrt(statistics.mean(r * r for r in residuals)))
            if residuals
            else None,
            "atm_iv_mean": float(statistics.mean(atm_iv_spread)) if atm_iv_spread else None,
            "atm_iv_std": float(statistics.pstdev(atm_iv_spread)) if len(atm_iv_spread) > 1 else None,
        }
        summary["days"][str(day)] = day_summary

        out_iv = OUT / f"iter0_iv_sample_day_{day}.csv"
        pd.DataFrame(iv_rows).head(5000).to_csv(out_iv, index=False)

    out_json = OUT / "iter0_iv_smile_spread_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
