#!/usr/bin/env python3
"""
Round-3 tape analysis for IV cross-strike ranking (r3v_cross_strike_iv_rank_07).

Timing (authoritative): round3work/round3description.txt + intraday winding used in
round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py:
  - CSV historical day column `day` in tapes: 0 -> 8 DTE at open, 1 -> 7, 2 -> 6.
  - Intraday: dte_eff = dte_open - (timestamp//100)/10000 (one day across session).
  - T (years) = dte_eff / 365, r = 0.

Core: Black–Scholes implied vol from voucher mid vs VELVETFRUIT_EXTRACT mid; vega at IV.
Supporting: spread (ask-bid), neighbor IV gaps |IV_i - IV_{i±1}|.

Writes JSON consumed by analysis.json (paths relative to repo root).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
OUT_JSON = OUT_DIR / "iv_rank_summary.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
STEP = 50  # subsample timestamps for speed


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
    fl, fh = f(lo), f(hi)
    if fl > 0 or fh < 0:
        return float("nan")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(S * float(norm.pdf(d1)) * math.sqrt(T))


def load_day_wide(day: int) -> pd.DataFrame:
    path = DATA / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
    if "VELVETFRUIT_EXTRACT" not in pvt.columns:
        raise RuntimeError("missing underlying")
    vcols = [v for v in VOUCHERS if v in pvt.columns]
    out = pvt[["VELVETFRUIT_EXTRACT"] + vcols].copy()
    out.columns = ["S"] + vcols
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_iv_spreads: list[float] = []
    neighbor_gaps: list[float] = []
    tick_used = 0
    vega_samples: list[float] = []
    voucher_spreads: list[float] = []
    # For decile-style (2 lowest / 2 highest IV among valid strikes): would neighbor cap allow?
    caps = [0.06, 0.09, 0.12, 0.15]
    pass_low = {c: 0 for c in caps}
    pass_high = {c: 0 for c in caps}
    for day in (0, 1, 2):
        wide = load_day_wide(day).sort_index()
        raw = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        idx = wide.index.to_numpy()
        take = np.unique(np.r_[0, len(idx) - 1, np.arange(0, len(idx), STEP, dtype=int)])
        for pos in take:
            ts = int(idx[pos])
            row = wide.iloc[pos]
            S = float(row["S"])
            if S <= 0:
                continue
            T = t_years_effective(day, ts)
            if T <= 0:
                continue
            ivs: list[float] = []
            for v in VOUCHERS:
                if v not in row.index:
                    continue
                mid = float(row[v])
                K = float(v.split("_")[1])
                iv = implied_vol_call(mid, S, K, T, 0.0)
                if not np.isfinite(iv):
                    ivs.append(float("nan"))
                    continue
                ivs.append(iv)
                vega_samples.append(bs_vega(S, K, T, iv, 0.0))

            arr = np.asarray(ivs, dtype=float)
            n = len(arr)
            valid_idx = [i for i in range(n) if np.isfinite(arr[i])]
            if len(valid_idx) < 8:
                continue
            tick_used += 1
            valid_idx.sort(key=lambda i: float(arr[i]))
            low_i, low_j = valid_idx[0], valid_idx[1]
            high_i, high_j = valid_idx[-2], valid_idx[-1]

            def max_gap_for_index(i: int) -> float:
                g: list[float] = []
                for j in (i - 1, i + 1):
                    if 0 <= j < n and np.isfinite(arr[i]) and np.isfinite(arr[j]):
                        g.append(abs(float(arr[i] - arr[j])))
                return max(g) if g else 0.0

            for c in caps:
                if max_gap_for_index(low_i) <= c and max_gap_for_index(low_j) <= c:
                    pass_low[c] += 1
                if max_gap_for_index(high_i) <= c and max_gap_for_index(high_j) <= c:
                    pass_high[c] += 1

            for i in range(1, n):
                if np.isfinite(arr[i]) and np.isfinite(arr[i - 1]):
                    neighbor_gaps.append(abs(float(arr[i] - arr[i - 1])))

            all_iv_spreads.append(float(np.nanmax(arr) - np.nanmin(arr)))

            sub = raw[(raw["timestamp"] == ts) & (raw["product"].isin(VOUCHERS))]
            for _, r in sub.iterrows():
                bp = r.get("bid_price_1")
                ap = r.get("ask_price_1")
                if pd.notna(bp) and pd.notna(ap):
                    voucher_spreads.append(float(ap) - float(bp))

    rel_out = str(OUT_JSON.relative_to(REPO))
    payload = {
        "timing_assumptions": {
            "source": "round3work/round3description.txt + plot_iv_smile_round3 intraday winding",
            "csv_day_to_dte_at_open": {"0": 8, "1": 7, "2": 6},
            "dte_effective_formula": "max(8 - csv_day - (timestamp//100)/10000, 1e-6)",
            "T_years": "dte_effective / 365",
            "r": 0.0,
        },
        "iv_method": "Bisection root of BS European call vs mid (bracket 1e-5..15, 50 iters; same no-arbitrage guards as plot_iv_smile_round3)",
        "greeks": "vega at solved IV (BS analytical)",
        "subsample_step": STEP,
        "iv_range_across_strikes": {
            "mean_smile_width": float(np.nanmean(all_iv_spreads)) if all_iv_spreads else None,
            "p50": float(np.nanpercentile(all_iv_spreads, 50)) if all_iv_spreads else None,
        },
        "neighbor_iv_gap_abs": {
            "mean": float(np.mean(neighbor_gaps)) if neighbor_gaps else None,
            "p90": float(np.percentile(neighbor_gaps, 90)) if neighbor_gaps else None,
        },
        "vega_at_iv": {
            "median": float(np.median(vega_samples)) if vega_samples else None,
            "p10": float(np.percentile(vega_samples, 10)) if vega_samples else None,
        },
        "voucher_bid_ask_spread_level1": {
            "mean": float(np.mean(voucher_spreads)) if voucher_spreads else None,
            "p90": float(np.percentile(voucher_spreads, 90)) if voucher_spreads else None,
        },
        "ticks_with_at_least_8_finite_ivs": tick_used,
        "neighbor_cap_sweep_decile_pairs": {
            "eligible_ticks": tick_used,
            "fraction_low_decile_tradable": {str(c): (pass_low[c] / tick_used if tick_used else None) for c in caps},
            "fraction_high_decile_tradable": {str(c): (pass_high[c] / tick_used if tick_used else None) for c in caps},
        },
        "output_file": rel_out,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)
    print(json.dumps({k: payload[k] for k in ("iv_range_across_strikes", "neighbor_iv_gap_abs", "vega_at_iv")}, indent=2))


if __name__ == "__main__":
    main()
