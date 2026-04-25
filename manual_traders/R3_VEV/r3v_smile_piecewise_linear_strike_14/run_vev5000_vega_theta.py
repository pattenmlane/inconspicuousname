#!/usr/bin/env python3
"""
Round 3 tapes: VEV_5000 — implied IV from mid, BS vega and theta (r=0), T=t_years_effective(day, ts).
Output analysis_outputs/vev5000_vega_theta_stats.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "vev5000_vega_theta_stats.json"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _r3v_smile_core import implied_vol_bisect, t_years_effective, bs_vega, bs_call_theta  # noqa: E402

U = "VELVETFRUIT_EXTRACT"
OPT = "VEV_5000"
K0 = 5000


def main() -> None:
    out: dict = {"by_day": {}}
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        if U not in pvt.columns or OPT not in pvt.columns:
            continue
        vegas: list[float] = []
        thetas: list[float] = []
        for ts, row in pvt.iterrows():
            S = float(row[U])
            mid = float(row[OPT])
            if not (S > 0 and mid > 0):
                continue
            T = t_years_effective(day, int(ts))
            iv = implied_vol_bisect(mid, S, float(K0), T)
            if iv is None or not np.isfinite(iv):
                continue
            vg = bs_vega(S, float(K0), T, iv)
            th = bs_call_theta(S, float(K0), T, iv)
            if np.isfinite(vg) and np.isfinite(th):
                vegas.append(float(vg))
                thetas.append(float(th))
        if not vegas:
            continue
        arr_v = np.asarray(vegas)
        arr_t = np.asarray(thetas)
        out["by_day"][str(day)] = {
            "n": int(len(arr_v)),
            "vega_median": float(np.median(arr_v)),
            "vega_p10_p90": [float(np.quantile(arr_v, 0.1)), float(np.quantile(arr_v, 0.9))],
            "theta_median": float(np.median(arr_t)),
            "theta_p10_p90": [float(np.quantile(arr_t, 0.1)), float(np.quantile(arr_t, 0.9))],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    full = {
        "method": f"IV from mid (bisect, r=0); T=t_years_effective; vega/theta from _r3v_smile_core for {OPT} vs {U} mid.",
        "stats": out,
    }
    OUT.write_text(json.dumps(full, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
