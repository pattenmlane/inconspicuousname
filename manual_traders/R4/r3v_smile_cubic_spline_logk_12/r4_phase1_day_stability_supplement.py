#!/usr/bin/env python3
"""
Round 4 — Phase 1 **day-stability** supplement (no re-scan of tapes).

Reads analysis_outputs/r4_trade_markouts_wide.csv from Phase 1 and emits small CSVs
with per-day n / mean / std for headline counterparty × aggressor × symbol cells.

Run: python3 manual_traders/R4/r3v_smile_cubic_spline_logk_12/r4_phase1_day_stability_supplement.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs"
EX = "VELVETFRUIT_EXTRACT"
HY = "HYDROGEL_PACK"


def agg_day(df: pd.DataFrame, col: str = "mark_20_same") -> pd.DataFrame:
    g = df.groupby("day")[col].agg(n="count", mean="mean", std="std").reset_index()
    g["col"] = col
    return g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(OUT / "r4_trade_markouts_wide.csv")
    wide["day"] = wide["day"].astype(int)

    rows: list[pd.DataFrame] = []

    # Top Phase-1 extract pairs (buy-aggr)
    ba = wide[(wide["aggressor"] == "buy_aggr") & (wide["symbol"] == EX)]

    for buyer, seller, name in [
        ("Mark 67", "Mark 49", "m67_m49"),
        ("Mark 67", "Mark 22", "m67_m22"),
    ]:
        sub = ba[(ba["buyer"] == buyer) & (ba["seller"] == seller)]
        g = agg_day(sub)
        g.insert(0, "slice", name)
        rows.append(g)

    # Phase-1 rank 3: Mark 14 as seller (buy-aggr extract, any buyer)
    sub = ba[ba["seller"] == "Mark 14"]
    g = agg_day(sub)
    g.insert(0, "slice", "buyaggr_seller_m14")
    rows.append(g)

    # Mark 22 / 49 as seller (buy-aggr extract) — adverse / drift
    for seller, tag in [("Mark 22", "seller22"), ("Mark 49", "seller49")]:
        sub = ba[ba["seller"] == seller]
        g = agg_day(sub)
        g.insert(0, "slice", f"buyaggr_{tag}")
        rows.append(g)

    # Hydro: Mark 38 -> Mark 14 (any aggressor — match pair table)
    hy = wide[(wide["symbol"] == HY) & (wide["buyer"] == "Mark 38") & (wide["seller"] == "Mark 14")]
    g = agg_day(hy)
    g.insert(0, "slice", "hydro_m38_m14")
    rows.append(g)

    # Horizons for Mark67→Mark49 buy-aggr extract (K=5,20,100 same-symbol)
    sub = ba[(ba["buyer"] == "Mark 67") & (ba["seller"] == "Mark 49")]
    for col, tag in [
        ("mark_5_same", "m5"),
        ("mark_20_same", "m20"),
        ("mark_100_same", "m100"),
    ]:
        g = agg_day(sub, col)
        g.insert(0, "slice", f"m67_m49_{tag}")
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "r4_phase1_top_edges_day_stability_k20.csv", index=False)

    # Pooled vs per-day sign consistency table (text)
    lines = ["=== Phase-1 headline cells: per-day mean (extract buy_aggr unless noted) ===\n"]
    for slice_name in out["slice"].unique():
        sub = out[out["slice"] == slice_name].sort_values("day")
        col = sub["col"].iloc[0]
        means = sub["mean"].tolist()
        ns = sub["n"].astype(int).tolist()
        if slice_name == "hydro_m38_m14":
            stab = sum(1 for m in means if m < 0)
            stab_lbl = f"negative_mean_days={stab}/3"
        else:
            stab = sum(1 for m in means if m > 0)
            stab_lbl = f"positive_mean_days={stab}/3"
        lines.append(
            f"{slice_name} [{col}]: days n={dict(zip(sub['day'], ns))} "
            f"mean={dict(zip(sub['day'], [round(m, 4) for m in means]))}  {stab_lbl}\n"
        )

    pooled = []
    for slice_name in ["m67_m49", "m67_m22", "buyaggr_seller22", "buyaggr_seller49"]:
        sub = wide[(wide["aggressor"] == "buy_aggr") & (wide["symbol"] == EX)]
        if slice_name == "m67_m49":
            sub = sub[(sub["buyer"] == "Mark 67") & (sub["seller"] == "Mark 49")]
        elif slice_name == "m67_m22":
            sub = sub[(sub["buyer"] == "Mark 67") & (sub["seller"] == "Mark 22")]
        elif slice_name == "buyaggr_seller22":
            sub = sub[sub["seller"] == "Mark 22"]
        else:
            sub = sub[sub["seller"] == "Mark 49"]
        pooled.append(
            f"{slice_name}: n={len(sub)} mean_m20={sub['mark_20_same'].mean():.4f} std={sub['mark_20_same'].std():.4f}"
        )
    lines.append("\nPooled (all days):\n" + "\n".join(pooled) + "\n")

    (OUT / "r4_phase1_top_edges_day_stability_summary.txt").write_text("".join(lines), encoding="utf-8")
    print("Wrote", OUT / "r4_phase1_top_edges_day_stability_k20.csv")
    print("Wrote", OUT / "r4_phase1_top_edges_day_stability_summary.txt")


if __name__ == "__main__":
    main()
