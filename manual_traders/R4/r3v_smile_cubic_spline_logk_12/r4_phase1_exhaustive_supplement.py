#!/usr/bin/env python3
"""
Round 4 Phase 1 — exhaustive supplement (reads Phase-1 wide CSV + graph; no tape re-parse).

Outputs under analysis_outputs/:
  r4_phase1_mark_coverage.txt          — distinct Marks, row counts, buyer/seller share
  r4_phase1_top_cells_bootstrap_k20.csv — percentile bootstrap CIs for headline mark_20_same cells
  r4_phase1_graph_2hop_motifs.csv      — buyer→seller→seller two-hop counts from graph edges
  r4_phase1_graph_triads.csv           — directed 3-cycles A→B→C→A (if any)
  r4_phase1_extract_buyaggr_hour_spread_quintile_m67sellers_k20.csv — stratification table

Run: python3 manual_traders/R4/r3v_smile_cubic_spline_logk_12/r4_phase1_exhaustive_supplement.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "analysis_outputs"
WIDE = OUT / "r4_trade_markouts_wide.csv"
EDGES = OUT / "r4_graph_edges.csv"
EX = "VELVETFRUIT_EXTRACT"
HY = "HYDROGEL_PACK"
RNG = np.random.default_rng(42)
N_BOOT = 4000


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float(np.nanmean(x)), float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = float(np.mean(RNG.choice(x, size=n, replace=True)))
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(WIDE)
    edges = pd.read_csv(EDGES)

    # --- Mark coverage ---
    marks = sorted(
        set(wide["buyer"].dropna().astype(str).unique()) | set(wide["seller"].dropna().astype(str).unique())
    )
    lines = [
        "Distinct buyer/seller names in r4_trade_markouts_wide.csv:",
        ", ".join(marks),
        f"count = {len(marks)}",
        "",
        "Row counts (all products):",
    ]
    for m in marks:
        nb = int((wide["buyer"] == m).sum())
        ns = int((wide["seller"] == m).sum())
        lines.append(f"  {m}: as_buyer={nb} as_seller={ns} total_touch={nb + ns}")
    (OUT / "r4_phase1_mark_coverage.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- Bootstrap on headline mark_20_same subsets ---
    col = "mark_20_same"
    rows = []
    wex = wide[wide["symbol"] == EX]

    def add_row(name: str, mask: pd.Series):
        mask = mask.reindex(wex.index, fill_value=False).astype(bool)
        sub = wex.loc[mask, col].astype(float).to_numpy()
        m, lo, hi = bootstrap_mean_ci(sub)
        rows.append(
            {
                "slice": name,
                "n": int(mask.sum()),
                "mean": m,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )

    ba = wex["aggressor"] == "buy_aggr"
    add_row("extract_buyaggr_seller22", ba & (wex["seller"] == "Mark 22"))
    add_row("extract_buyaggr_seller49", ba & (wex["seller"] == "Mark 49"))
    add_row("extract_buyaggr_seller14", ba & (wex["seller"] == "Mark 14"))
    add_row("extract_buyaggr_buyer67", ba & (wex["buyer"] == "Mark 67"))
    add_row("extract_buyaggr_m67_m49", ba & (wex["buyer"] == "Mark 67") & (wex["seller"] == "Mark 49"))
    add_row("extract_buyaggr_m67_m22", ba & (wex["buyer"] == "Mark 67") & (wex["seller"] == "Mark 22"))
    why = wide[(wide["symbol"] == HY) & (wide["buyer"] == "Mark 38") & (wide["seller"] == "Mark 14")]
    m, lo, hi = bootstrap_mean_ci(why[col].astype(float).to_numpy())
    rows.append(
        {
            "slice": "hydro_m38_m14_allaggr",
            "n": len(why),
            "mean": m,
            "ci95_low": lo,
            "ci95_high": hi,
        }
    )
    pd.DataFrame(rows).to_csv(OUT / "r4_phase1_top_cells_bootstrap_k20.csv", index=False)

    # --- 2-hop motifs from static edge counts (structural choreography proxy) ---
    nmat: dict[tuple[str, str], int] = {}
    for _, r in edges.iterrows():
        nmat[(str(r["buyer"]), str(r["seller"]))] = int(r["n"])
    nodes = sorted({a for (a, _) in nmat} | {b for (_, b) in nmat})
    hops = []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            n_ab = nmat.get((a, b), 0)
            if n_ab == 0:
                continue
            for c in nodes:
                if c in (a, b):
                    continue
                n_bc = nmat.get((b, c), 0)
                if n_bc == 0:
                    continue
                hops.append({"A": a, "B": b, "C": c, "n_AB": n_ab, "n_BC": n_bc, "min_n": min(n_ab, n_bc)})
    hop_df = pd.DataFrame(hops).sort_values("min_n", ascending=False)
    hop_df.to_csv(OUT / "r4_phase1_graph_2hop_motifs.csv", index=False)

    triads = []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            if nmat.get((a, b), 0) == 0:
                continue
            for c in nodes:
                if c in (a, b):
                    continue
                if nmat.get((b, c), 0) == 0 or nmat.get((c, a), 0) == 0:
                    continue
                triads.append(
                    {
                        "cycle": f"{a}->{b}->{c}->{a}",
                        "n_AB": nmat[(a, b)],
                        "n_BC": nmat[(b, c)],
                        "n_CA": nmat[(c, a)],
                    }
                )
    pd.DataFrame(triads).drop_duplicates().to_csv(OUT / "r4_phase1_graph_triads.csv", index=False)

    # --- Hour × spread quintile for M67→49 / M67→22 extract buy-aggr K=20 ---
    sub = wex[ba & (wex["buyer"] == "Mark 67") & (wex["seller"].isin(["Mark 49", "Mark 22"]))].copy()
    hour_path = OUT / "r4_phase1_extract_buyaggr_hour_spread_quintile_m67sellers_k20.csv"
    if len(sub) >= 25:
        sub["hour_bin"] = pd.cut(
            sub["hour"].astype(float),
            bins=[-0.1, 6, 12, 18, 24.1],
            labels=["0-6", "6-12", "12-18", "18-24"],
        )
        spr = sub["spr_symbol"].astype(float)
        sub["spr_q"] = pd.qcut(spr.rank(method="first"), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        g = (
            sub.groupby(["seller", "hour_bin", "spr_q"], observed=False)[col]
            .agg(n="count", mean="mean", median="median")
            .reset_index()
        )
        g.to_csv(hour_path, index=False)
    else:
        hour_path.write_text("insufficient_rows_for_stratification\n", encoding="utf-8")

    print("Wrote", OUT / "r4_phase1_mark_coverage.txt")
    print("Wrote", OUT / "r4_phase1_top_cells_bootstrap_k20.csv")
    print("Wrote", OUT / "r4_phase1_graph_2hop_motifs.csv")
    print("Wrote", OUT / "r4_phase1_graph_triads.csv")
    print("Wrote", hour_path)


if __name__ == "__main__":
    main()
