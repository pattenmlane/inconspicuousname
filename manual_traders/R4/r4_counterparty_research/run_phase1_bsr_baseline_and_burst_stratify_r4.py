#!/usr/bin/env python3
"""
Phase 1 supplement (bullet 2 + optional burst stratify for bullet 1):

1) **Leave-one-out** baseline within (buyer, seller, symbol, spr_regime): for each print,
   baseline = mean of mark_20_u over **other** rows in the same cell (so residuals are not
   degenerate like plain cell-mean subtraction). Rows with cell_n==1 get NaN baseline.
   Export full CSV + top |residual| rows and aggregated cell summaries.

2) Merge burst flag (>=4 prints same day+timestamp). For high-volume aggressor roles,
   report mean mark_20_u and mark_20_sym split burst vs non, pooled and by day (min n>=15 per slice).

Input: r4_trades_enriched_markouts.csv + raw trades for burst counts.
Output:
  r4_trades_with_resid_baseline_bsr.csv
  r4_phase1_residual_bsr_top_abs_rows.csv
  r4_phase1_residual_bsr_cell_summary.csv
  r4_phase1_aggressor_mark20_by_burst_pooled.csv
  r4_phase1_aggressor_mark20_by_burst_by_day.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
ENR = Path(__file__).resolve().parent / "outputs" / "r4_trades_enriched_markouts.csv"
OUT = Path(__file__).resolve().parent / "outputs"
MIN_CELL = 15


def burst_frame() -> pd.DataFrame:
    parts = []
    for d in (1, 2, 3):
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        parts.append(t)
    tr = pd.concat(parts, ignore_index=True)
    cnt = tr.groupby(["day", "timestamp"]).size().rename("n_prints").reset_index()
    cnt["burst"] = cnt["n_prints"] >= 4
    return cnt[["day", "timestamp", "burst"]]


def main() -> None:
    enr = pd.read_csv(ENR)
    for c in ("buyer", "seller", "symbol", "spr_regime", "aggressor"):
        enr[c] = enr[c].astype(str)
    enr["mark_20_u"] = pd.to_numeric(enr["mark_20_u"], errors="coerce")
    enr["mark_20_sym"] = pd.to_numeric(enr["mark_20_sym"], errors="coerce")

    gcols = ["buyer", "seller", "symbol", "spr_regime"]
    enr["_u"] = enr["mark_20_u"]
    enr["_sum"] = enr.groupby(gcols)["_u"].transform("sum")
    enr["_cnt"] = enr.groupby(gcols)["_u"].transform("count")
    enr["baseline_loo_u20"] = np.where(
        enr["_cnt"] > 1,
        (enr["_sum"] - enr["_u"]) / (enr["_cnt"] - 1),
        np.nan,
    )
    enr["resid_u20_bsr_loo"] = enr["_u"] - enr["baseline_loo_u20"]
    enr = enr.drop(columns=["_u", "_sum", "_cnt"])
    OUT.mkdir(parents=True, exist_ok=True)
    enr.to_csv(OUT / "r4_trades_with_resid_baseline_bsr.csv", index=False)

    rr = enr[enr["resid_u20_bsr_loo"].notna()].copy()
    rr["abs_resid"] = rr["resid_u20_bsr_loo"].abs()
    top = rr.nlargest(80, "abs_resid")[
        [
            "day",
            "timestamp",
            "buyer",
            "seller",
            "symbol",
            "spr_regime",
            "mark_20_u",
            "baseline_loo_u20",
            "resid_u20_bsr_loo",
        ]
    ]
    top.to_csv(OUT / "r4_phase1_residual_bsr_top_abs_rows.csv", index=False)

    summ = (
        rr.groupby(gcols)
        .agg(
            n=("resid_u20_bsr_loo", "count"),
            mean_resid=("resid_u20_bsr_loo", "mean"),
            std_resid=("resid_u20_bsr_loo", "std"),
            mean_u20=("mark_20_u", "mean"),
        )
        .reset_index()
    )
    summ = summ[summ["n"] >= MIN_CELL].sort_values("std_resid", ascending=False)
    summ.to_csv(OUT / "r4_phase1_residual_bsr_cell_summary.csv", index=False)

    bf = burst_frame()
    mb = enr.merge(bf, on=["day", "timestamp"], how="left")
    mb["burst"] = mb["burst"].fillna(False)

    slices = [
        ("aggr_buy", "Mark 67", "buyer", mb["aggressor"] == "buy"),
        ("aggr_buy", "Mark 55", "buyer", mb["aggressor"] == "buy"),
        ("aggr_buy", "Mark 38", "buyer", mb["aggressor"] == "buy"),
        ("aggr_sell", "Mark 22", "seller", mb["aggressor"] == "sell"),
    ]
    pooled_rows = []
    day_rows = []
    for role, name, col, mask0 in slices:
        sub = mb[mask0 & (mb[col] == name)]
        for burst, lab in [(True, "burst"), (False, "non_burst")]:
            g = sub[sub["burst"] == burst]
            u = g["mark_20_u"].dropna()
            sy = g["mark_20_sym"].dropna()
            if len(u) < MIN_CELL:
                continue
            pooled_rows.append(
                {
                    "role": role,
                    "name": name,
                    "burst": lab,
                    "n": int(len(g)),
                    "n_u20": int(len(u)),
                    "mean_u20": float(u.mean()),
                    "mean_sym20": float(sy.mean()) if len(sy) else float("nan"),
                }
            )
        for d, gd in sub.groupby("day"):
            for burst, lab in [(True, "burst"), (False, "non_burst")]:
                g = gd[gd["burst"] == burst]
                u = g["mark_20_u"].dropna()
                sy = g["mark_20_sym"].dropna()
                if len(u) < 10:
                    continue
                day_rows.append(
                    {
                        "role": role,
                        "name": name,
                        "day": int(d),
                        "burst": lab,
                        "n_u20": int(len(u)),
                        "mean_u20": float(u.mean()),
                        "mean_sym20": float(sy.mean()) if len(sy) else float("nan"),
                    }
                )

    pd.DataFrame(pooled_rows).to_csv(OUT / "r4_phase1_aggressor_mark20_by_burst_pooled.csv", index=False)
    pd.DataFrame(day_rows).to_csv(OUT / "r4_phase1_aggressor_mark20_by_burst_by_day.csv", index=False)
    print("wrote BSR residual + burst stratify outputs to", OUT)


if __name__ == "__main__":
    main()
