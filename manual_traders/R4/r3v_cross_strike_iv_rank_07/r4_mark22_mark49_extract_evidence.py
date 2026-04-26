#!/usr/bin/env python3
"""
Phase 1 adverse-selection slice (r4_p1_trade_enriched.csv): Mark 22 seller, Mark 49 buyer,
VELVETFRUIT_EXTRACT — horizons k∈{5,20,100}, sell_agg-only vs all rows, by day.

Outputs:
- r4_p1_m22_m49_extract_markout_by_day.csv
- r4_p1_m22_m49_extract_markout_summary.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "analysis_outputs", "r4_p1_trade_enriched.csv")
OUT_CSV = os.path.join(HERE, "analysis_outputs", "r4_p1_m22_m49_extract_markout_by_day.csv")
OUT_JSON = os.path.join(HERE, "analysis_outputs", "r4_p1_m22_m49_extract_markout_summary.json")


def stats(x: pd.Series) -> dict:
    v = x.dropna()
    n = int(len(v))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "t": float("nan")}
    m = float(v.mean())
    s = float(v.std(ddof=1)) if n > 1 else float("nan")
    t = m / (s / np.sqrt(n)) if s == s and s > 1e-12 else float("nan")
    return {"n": n, "mean": m, "std": s, "t": t}


def main() -> None:
    df = pd.read_csv(SRC)
    m = df[
        (df["seller"] == "Mark 22")
        & (df["buyer"] == "Mark 49")
        & (df["symbol"] == "VELVETFRUIT_EXTRACT")
    ].copy()

    rows = []
    for k in (5, 20, 100):
        col = f"dm_self_k{k}"
        rows.append({"slice": "all_rows", "day": -1, "horizon_k": k, **stats(m[col])})
        rows.append({"slice": "sell_agg_only", "day": -1, "horizon_k": k, **stats(m.loc[m["agg"] == "sell_agg", col])})
        for d, g in m.groupby("day"):
            rows.append({"slice": "all_rows", "day": int(d), "horizon_k": k, **stats(g[col])})
            rows.append({"slice": "sell_agg_only", "day": int(d), "horizon_k": k, **stats(g.loc[g["agg"] == "sell_agg", col])})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    sell_only = m[m["agg"] == "sell_agg"]
    summary = {
        "n_all": int(len(m)),
        "n_sell_agg": int(len(sell_only)),
        "note": "Mark 22 seller + Mark 49 buyer on extract; sell_agg means trade price <= L1 bid (Mark 22 aggressive sell).",
        "pooled_sell_agg": {
            "k5": stats(sell_only["dm_self_k5"]),
            "k20": stats(sell_only["dm_self_k20"]),
            "k100": stats(sell_only["dm_self_k100"]),
        },
        "per_day_sell_agg_k20": {
            int(d): stats(g["dm_self_k20"]) for d, g in sell_only.groupby("day")
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
