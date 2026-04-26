#!/usr/bin/env python3
"""
Phase 1 bullet 3 supplement: directed graph metrics — hubs, in/out notional, reciprocity of top pairs.
Inputs: Prosperity4Data/ROUND_4/trades_round_4_day_*.csv
Outputs: r4_phase1_graph_hubs_by_name.csv, r4_phase1_graph_pair_reciprocity.csv, r4_phase1_graph_reciprocity_top.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    parts = []
    for d in (1, 2, 3):
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        parts.append(t)
    tr = pd.concat(parts, ignore_index=True)
    tr["notional"] = tr["price"].astype(float) * tr["quantity"].astype(float)
    g = tr.groupby(["buyer", "seller"], as_index=False).agg(
        n_prints=("symbol", "count"), notional=("notional", "sum")
    )

    all_names = set(tr["buyer"].astype(str).unique()) | set(tr["seller"].astype(str).unique())
    rows = []
    for n in sorted(all_names):
        if n in ("nan", "None", ""):
            continue
        gb = g[g["buyer"] == n]
        gs = g[g["seller"] == n]
        rows.append(
            {
                "name": n,
                "n_distinct_counterparties_as_buyer": int(len(gb)),
                "n_distinct_counterparties_as_seller": int(len(gs)),
                "n_prints_as_buyer": int(gb["n_prints"].sum()) if len(gb) else 0,
                "n_prints_as_seller": int(gs["n_prints"].sum()) if len(gs) else 0,
                "notional_as_buyer": float(gb["notional"].sum()) if len(gb) else 0.0,
                "notional_as_seller": float(gs["notional"].sum()) if len(gs) else 0.0,
            }
        )
    hub = pd.DataFrame(rows)
    hub["net_notional_buyer_minus_seller"] = hub["notional_as_buyer"] - hub["notional_as_seller"]
    hub = hub.sort_values("n_prints_as_buyer", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    hub.to_csv(OUT / "r4_phase1_graph_hubs_by_name.csv", index=False)

    top = g[g["n_prints"] >= 5].copy()
    recip = []
    for _, r in top.iterrows():
        b, s, n, no = r["buyer"], r["seller"], int(r["n_prints"]), float(r["notional"])
        rev = g[(g["buyer"] == s) & (g["seller"] == b)]
        recip.append(
            {
                "buyer": b,
                "seller": s,
                "n_prints": n,
                "notional": no,
                "reciprocal_n_prints": int(rev["n_prints"].iloc[0]) if len(rev) else 0,
                "reciprocal_notional": float(rev["notional"].iloc[0]) if len(rev) else 0.0,
            }
        )
    rec_df = pd.DataFrame(recip).sort_values("n_prints", ascending=False)
    rec_df.to_csv(OUT / "r4_phase1_graph_pair_reciprocity.csv", index=False)
    top_rec = rec_df[rec_df["reciprocal_n_prints"] > 0].head(25).to_dict(orient="records")
    (OUT / "r4_phase1_graph_reciprocity_top.json").write_text(
        json.dumps({"note": "Pairs with n>=5 directed prints; reverse edge if exists", "top": top_rec}, indent=2),
        encoding="utf-8",
    )
    print("wrote", OUT / "r4_phase1_graph_hubs_by_name.csv", "n_names", len(hub))


if __name__ == "__main__":
    main()
