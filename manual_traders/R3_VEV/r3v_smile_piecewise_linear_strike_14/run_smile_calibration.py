#!/usr/bin/env python3
"""Offline calibration from ROUND_3 tapes → analysis_outputs/smile_cal_round3.json"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
KNOTS = (5000, 5200, 5400)
U = "VELVETFRUIT_EXTRACT"

# Import core from same package dir
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _r3v_smile_core import (  # noqa: E402
    infer_csv_day_from_smile,
    implied_vol_bisect,
    t_years_effective,
    fit_knot_ivs_least_squares,
    refine_knot_ivs_gauss_newton,
)


def load_wide(day: int) -> pd.DataFrame:
    p = DATA / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(p, sep=";")
    pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
    cols = [U] + [v for v in VOUCHERS if v in pvt.columns]
    return pvt[cols].copy()


def main() -> None:
    knot_acc: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    spread_stats: dict[str, list[float]] = defaultdict(list)
    iv_by_strike_day: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for csv_day in (0, 1, 2):
        wide = load_wide(csv_day)
        ts_list = list(wide.index)[::50]  # subsample
        for ts in ts_list:
            row = wide.loc[ts]
            S = float(row[U])
            if not math.isfinite(S) or S <= 0:
                continue
            mids = {int(k): float(row[f"VEV_{k}"]) for k in STRIKES if f"VEV_{k}" in row.index}
            mids = {k: v for k, v in mids.items() if math.isfinite(v)}
            if len(mids) < 6:
                continue
            d_inf = infer_csv_day_from_smile(S, mids, int(ts), STRIKES)
            T = t_years_effective(d_inf, int(ts))
            strikes = []
            ivs = []
            for K in STRIKES:
                mid = mids.get(K)
                if mid is None:
                    continue
                iv = implied_vol_bisect(mid, S, float(K), T)
                if iv is None:
                    continue
                strikes.append(K)
                ivs.append(iv)
                iv_by_strike_day[K][csv_day].append(iv)
            init = fit_knot_ivs_least_squares(strikes, ivs, KNOTS)
            if init is None:
                continue
            w = refine_knot_ivs_gauss_newton(strikes, ivs, init, KNOTS)
            knot_acc[csv_day].append(w)

        # spread proxy from raw CSV for one product (VEV_5200) same day
        raw = pd.read_csv(DATA / f"prices_round_3_day_{csv_day}.csv", sep=";")
        sub = raw[raw["product"] == "VEV_5200"]
        for _, r in sub.iloc[::200].iterrows():
            bp = r.get("bid_price_1")
            ap = r.get("ask_price_1")
            if pd.isna(bp) or pd.isna(ap):
                continue
            spread_stats["VEV_5200"].append(float(ap) - float(bp))

    cal: dict = {
        "timing": {
            "source": "round3work/round3description.txt + intraday wind from plot_iv_smile_round3",
            "csv_day_to_dte_open": {"0": 8, "1": 7, "2": 6},
            "dte_effective": "max(dte_open - (timestamp//100)/10000, 1e-6); T = dte_eff/365",
            "csv_day_inference": "minimize IV cross-strike variance vs T from candidate csv_day {0,1,2}",
        },
        "knot_strikes": list(KNOTS),
        "mean_knot_iv_by_csv_day": {},
        "mean_knot_iv_global": [0.0, 0.0, 0.0],
        "vev5200_median_spread_by_csv_day": {},
        "iv_median_by_strike_csv_day": {},
    }
    glob = [0.0, 0.0, 0.0]
    n_glob = 0
    for d, lst in knot_acc.items():
        if not lst:
            continue
        m0 = sum(x[0] for x in lst) / len(lst)
        m1 = sum(x[1] for x in lst) / len(lst)
        m2 = sum(x[2] for x in lst) / len(lst)
        cal["mean_knot_iv_by_csv_day"][str(d)] = [m0, m1, m2]
        glob[0] += m0 * len(lst)
        glob[1] += m1 * len(lst)
        glob[2] += m2 * len(lst)
        n_glob += len(lst)
        if spread_stats["VEV_5200"]:
            cal["vev5200_median_spread_by_csv_day"][str(d)] = float(
                sorted(spread_stats["VEV_5200"])[len(spread_stats["VEV_5200"]) // 2]
            )
    if n_glob:
        cal["mean_knot_iv_global"] = [glob[i] / n_glob for i in range(3)]

    for K in STRIKES:
        cal["iv_median_by_strike_csv_day"][str(K)] = {}
        for d in (0, 1, 2):
            arr = iv_by_strike_day[K][d]
            if arr:
                s = sorted(arr)
                cal["iv_median_by_strike_csv_day"][str(K)][str(d)] = float(s[len(s) // 2])

    out_path = OUT_DIR / "smile_cal_round3.json"
    out_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
