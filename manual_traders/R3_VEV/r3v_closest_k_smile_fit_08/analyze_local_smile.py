#!/usr/bin/env python3
"""Offline Round-3 tape analysis: local IV smile (closest strikes) vs full chain.

Uses the same BS/r=0/DTE conventions as round3work/plotting/.../plot_iv_smile_round3.py
(CSV day 0->DTE 8 at open, intraday wind-down).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
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
        fl, fh = f(lo), f(hi)
        if fl > 0 or fh < 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-8, rtol=1e-8)
    except ValueError:
        return float("nan")


def load_day_wide(day: int) -> pd.DataFrame:
    path = DATA / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
    vcols = [v for v in VOUCHERS if v in pvt.columns]
    out = pvt[["VELVETFRUIT_EXTRACT"] + vcols].copy()
    out.columns = ["S"] + vcols
    return out


def local_iv_rmse_row(row: pd.Series, day: int, k_closest: int) -> tuple[float, float]:
    """Return (rmse_local_quad, rmse_flat_atm) for one timestamp."""
    S = float(row["S"])
    T = t_years_effective(day, int(row.name))
    ivs: dict[str, float] = {}
    for v in VOUCHERS:
        K = int(v.split("_")[1])
        mkt = float(row[v])
        ivs[v] = implied_vol_call(mkt, S, K, T, 0.0)

    valid = [(int(v.split("_")[1]), ivs[v]) for v in VOUCHERS if not math.isnan(ivs[v])]
    if len(valid) < 3:
        return float("nan"), float("nan")

    strikes_all = np.array([a[0] for a in valid])
    iv_all = np.array([a[1] for a in valid])
    dist = np.abs(strikes_all - S)
    order = np.argsort(dist)[:k_closest]
    xs = np.log(strikes_all[order] / S)
    ys = iv_all[order]
    coef = np.polyfit(xs, ys, deg=min(2, len(ys) - 1))

    def pred_iv(K: float) -> float:
        x = math.log(K / S)
        return float(np.polyval(coef, x))

    sq_local = []
    sq_flat = []
    atm_iv = float(np.nanmedian(iv_all))
    for K, iv in zip(strikes_all, iv_all):
        sq_local.append((pred_iv(float(K)) - iv) ** 2)
        sq_flat.append((atm_iv - iv) ** 2)
    return math.sqrt(float(np.mean(sq_local))), math.sqrt(float(np.mean(sq_flat)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary: dict = {"days": {}, "method": __doc__}
    for day in (0, 1, 2):
        wide = load_day_wide(day)
        idx = wide.index.to_numpy()
        sample = sorted(rng.choice(idx, size=min(500, len(idx)), replace=False).tolist())
        rows_local_k3 = []
        rows_local_k6 = []
        for ts in sample:
            r = wide.loc[ts]
            rows_local_k3.append(local_iv_rmse_row(r, day, 3))
            rows_local_k6.append(local_iv_rmse_row(r, day, 6))
        a3 = np.array(rows_local_k3, dtype=float)
        a6 = np.array(rows_local_k6, dtype=float)
        summary["days"][str(day)] = {
            "n_samples": len(sample),
            "median_rmse_iv_k3_vs_mkt": float(np.nanmedian(a3[:, 0])),
            "median_rmse_iv_k6_vs_mkt": float(np.nanmedian(a6[:, 0])),
            "median_rmse_flat_atm_vs_mkt": float(np.nanmedian(a3[:, 1])),
            "spread_width_median_ticks": _spread_stats(day),
        }

    out_path = OUT_DIR / "local_smile_iv_rmse.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out_path)


def _spread_stats(day: int) -> float:
    path = DATA / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df[df["product"].isin(VOUCHERS + ["VELVETFRUIT_EXTRACT"])]
    spread = (
        df["ask_price_1"].astype(float) - df["bid_price_1"].astype(float)
    ).replace(0, np.nan)
    return float(spread.median())


if __name__ == "__main__":
    main()
