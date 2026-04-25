#!/usr/bin/env python3
"""
Round-3 tape analysis for r3v_jump_gap_filter_17: extract jumps, neighbor strike gaps,
implied vol / vega (Black–Scholes European call, r=0).

TTE / DTE: round3work/round3description.txt + intraday winding from
round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py
(dte_from_csv_day: csv day 0 -> 8d open, 1 -> 7d, 2 -> 6d; dte_eff winds ~1d per session).
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"))
from plot_iv_smile_round3 import implied_vol_call, t_years_effective  # noqa: E402

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
DATA_DIR = REPO / "Prosperity4Data" / "ROUND_3"
OUT_JSON = Path(__file__).resolve().parent / "analysis_jump_gap_stats.json"


def bs_call_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> tuple[float, float]:
    if T <= 0 or sigma <= 1e-12:
        return max(S - K, 0.0), 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - v)
    vega = S * float(norm.pdf(d1)) * math.sqrt(T)
    return float(price), float(vega)


def process_day(csv_day: int) -> dict:
    path = DATA_DIR / f"prices_round_3_day_{csv_day}.csv"
    df = pd.read_csv(path, sep=";")
    ex = df[df["product"] == "VELVETFRUIT_EXTRACT"].sort_values(["timestamp"])
    ts_list = ex["timestamp"].values
    s_mid = ex["mid_price"].values.astype(float)
    ds = np.abs(np.diff(s_mid, prepend=s_mid[0]))
    # jump proxy: step change in extract mid
    p95 = float(np.percentile(ds, 95))
    p99 = float(np.percentile(ds, 99))

    rows_out: list[dict] = []
    for i, t in enumerate(ts_list):
        S = float(s_mid[i])
        T = float(t_years_effective(csv_day, int(t)))
        if not math.isfinite(T) or T <= 0:
            continue
        sub = df[(df["timestamp"] == t) & df["product"].isin(VOUCHERS)]
        mids: dict[str, float] = {}
        for _, r in sub.iterrows():
            mids[str(r["product"])] = float(r["mid_price"])
        if len(mids) < len(VOUCHERS):
            continue
        theos: dict[str, float] = {}
        ivs: dict[str, float] = {}
        vegas: dict[str, float] = {}
        for k in STRIKES:
            sym = f"VEV_{k}"
            m = mids.get(sym)
            if m is None:
                continue
            iv = implied_vol_call(m, S, float(k), T, 0.0)
            if not (isinstance(iv, float) and math.isfinite(iv) and iv > 0):
                continue
            th, vg = bs_call_vega(S, float(k), T, iv, 0.0)
            theos[sym] = th
            ivs[sym] = iv
            vegas[sym] = vg
        if len(theos) < 4:
            continue
        atm_k = min(STRIKES, key=lambda kk: abs(float(kk) - S))
        ix = STRIKES.index(atm_k)
        gaps = []
        for j in range(max(0, ix - 2), min(len(STRIKES) - 1, ix + 2)):
            k0, k1 = STRIKES[j], STRIKES[j + 1]
            s0, s1 = f"VEV_{k0}", f"VEV_{k1}"
            if s0 not in mids or s1 not in mids or s0 not in theos or s1 not in theos:
                continue
            dm = mids[s1] - mids[s0]
            dt = theos[s1] - theos[s0]
            gaps.append(abs(dm - dt))
        neigh = float(np.mean(gaps)) if gaps else float("nan")
        rows_out.append(
            {
                "timestamp": int(t),
                "S": S,
                "dS_abs": float(ds[i]),
                "neighbor_abs_gap_mean": neigh,
                "iv_atm": ivs.get(f"VEV_{atm_k}", float("nan")),
                "vega_atm": vegas.get(f"VEV_{atm_k}", float("nan")),
            }
        )

    tbl = pd.DataFrame(rows_out)
    if tbl.empty:
        return {"csv_day": csv_day, "error": "no rows"}
    corr = float(tbl["dS_abs"].corr(tbl["neighbor_abs_gap_mean"])) if len(tbl) > 5 else float("nan")
    return {
        "csv_day": csv_day,
        "extract_dS_p95": p95,
        "extract_dS_p99": p99,
        "corr_abs_dS_neighbor_gap": corr,
        "neighbor_gap_median": float(tbl["neighbor_abs_gap_mean"].median()),
        "neighbor_gap_p90": float(tbl["neighbor_abs_gap_mean"].quantile(0.9)),
        "iv_atm_median": float(tbl["iv_atm"].median()) if tbl["iv_atm"].notna().any() else float("nan"),
        "sample_rows": len(tbl),
    }


def main() -> None:
    stats = []
    for d in (0, 1, 2):
        stats.append(process_day(d))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"days": stats, "method": "BS_call_IV_brentq_r0_T_from_plot_iv_smile_round3"}, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
