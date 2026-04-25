#!/usr/bin/env python3
"""
Round 3 tape analysis for r3v_velvet_beta_hedge_02: IV smile / vega-weighted IV,
spread stats, extract short-horizon moments.

TTE / DTE (authoritative: round3work/round3description.txt + combined_analysis convention):
- Historical CSV day column 0,1,2 maps to calendar DTE at session open 8,7,6 days.
- Intraday: DTE winds ~1 day over the session (same as plot_iv_smile_round3.t_years_effective).
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

# Replicate t_years_effective without importing plotting package
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
    if market <= intrinsic + 1e-9 or market >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
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


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(S * float(norm.pdf(d1)) * math.sqrt(T))


STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"


def load_day(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def main() -> None:
    rows_out: list[dict] = []
    spread_rows: list[dict] = []

    for csv_day in (0, 1, 2):
        p = DATA / f"prices_round_3_day_{csv_day}.csv"
        df = load_day(p)
        # Pivot mids by timestamp
        piv = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        if U not in piv.columns:
            print("missing extract", csv_day, file=sys.stderr)
            continue
        S = piv[U].astype(float)

        # Short-horizon extract stats (same grid as trader warmup)
        ret1 = S.diff().fillna(0.0)
        z20 = (S - S.rolling(20, min_periods=5).mean()) / (S.rolling(20, min_periods=5).std().replace(0, np.nan) + 1e-9)

        for v in VOUCHERS:
            if v not in piv.columns:
                continue
            sub = df[df["product"] == v].copy()
            for _, r in sub.iterrows():
                ts = int(r["timestamp"])
                T = t_years_effective(csv_day, ts)
                mid = float(r["mid_price"])
                K = int(v.split("_")[1])
                S_t = float(S.loc[ts]) if ts in S.index else float("nan")
                if not np.isfinite(S_t) or not np.isfinite(mid):
                    continue
                iv = implied_vol_call(mid, S_t, K, T, 0.0)
                vega = bs_vega(S_t, K, T, iv) if np.isfinite(iv) and iv > 0 else float("nan")
                bp = r.get("bid_price_1")
                ap = r.get("ask_price_1")
                spread = float(ap) - float(bp) if pd.notna(bp) and pd.notna(ap) else float("nan")

                rows_out.append(
                    {
                        "csv_day": csv_day,
                        "timestamp": ts,
                        "voucher": v,
                        "K": K,
                        "S": S_t,
                        "mid": mid,
                        "iv": iv,
                        "vega": vega,
                        "spread": spread,
                        "extract_ret1": float(ret1.loc[ts]) if ts in ret1.index else float("nan"),
                        "extract_z20": float(z20.loc[ts]) if ts in z20.index else float("nan"),
                    }
                )
                spread_rows.append({"csv_day": csv_day, "voucher": v, "spread": spread})

    tbl = pd.DataFrame(rows_out)
    # Keep repo small: stratified subsample for inspection (full stats use tbl in-memory).
    if len(tbl) > 0:
        sub = tbl.groupby(["csv_day", "voucher"], group_keys=False).apply(
            lambda g: g.iloc[:: max(1, len(g) // 50)], include_groups=False
        )
        sub.to_csv(OUT_DIR / "iv_vega_subsample.csv", index=False)

    # Vega-weighted mean IV by timestamp (ATM-ish weight: vega)
    finite = tbl[np.isfinite(tbl["iv"]) & np.isfinite(tbl["vega"])].copy()
    finite["w"] = finite["vega"].clip(lower=1e-6)

    def _vw_iv(gdf: pd.DataFrame) -> pd.Series:
        w = gdf["w"].values
        ivs = gdf["iv"].values
        return pd.Series({"iv_vega_mean": float(np.average(ivs, weights=w)), "vega_sum": float(gdf["vega"].sum())})

    g = finite.groupby(["csv_day", "timestamp"], group_keys=False).apply(_vw_iv, include_groups=False).reset_index()

    # Correlation IV_vega_mean change vs extract return (lead/lag 0)
    merge_parts = []
    for d in (0, 1, 2):
        p = DATA / f"prices_round_3_day_{d}.csv"
        df = load_day(p)
        piv = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        ex = piv[U].astype(float)
        merge_parts.append(
            pd.DataFrame(
                {
                    "csv_day": d,
                    "timestamp": ex.index,
                    "extract_ret1": ex.diff().fillna(0.0).values,
                }
            )
        )
    exr = pd.concat(merge_parts, ignore_index=True)
    mg = g.merge(exr, on=["csv_day", "timestamp"], how="inner")
    if len(mg) > 10:
        mg["div_change"] = mg.groupby("csv_day")["iv_vega_mean"].diff()
        corr = float(mg["div_change"].corr(mg["extract_ret1"]))
    else:
        corr = float("nan")

    summary = {
        "tte_convention": "CSV day d in {0,1,2} -> DTE_open = 8-d; intraday DTE_eff = DTE_open - (timestamp//100)/10000; T = DTE_eff/365 (see round3work/round3description.txt example + combined_analysis winding).",
        "iv_method": "Black-Scholes European call, r=0; IV from mid via brentq on [1e-5, 15]; intrinsic/deep ITM/invalid -> NaN.",
        "greeks": "Vega from BS at implied sigma; delta not exported in this pass but available via same formula as trader.",
        "vega_weighted_iv_vs_extract_ret_corr_diff": corr,
        "mean_spread_by_voucher": pd.DataFrame(spread_rows).groupby("voucher")["spread"].mean().to_dict(),
        "n_iv_rows_finite": int(np.isfinite(tbl["iv"]).sum()),
    }

    (OUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote iv_vega_subsample.csv, analysis_summary.json")


if __name__ == "__main__":
    main()
