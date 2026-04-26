#!/usr/bin/env python3
"""
Phase 1 follow-up: Mark 55 on VELVETFRUIT_EXTRACT — horizons k∈{5,20,100} and day-stability.

Uses r4_p1_trade_enriched.csv (buy_agg when buyer==M55, sell_agg when seller==M55).

Outputs:
- r4_p1_mark55_extract_by_day.csv — per day × side × horizon stats
- r4_p1_mark55_extract_summary.json — pooled + per-day sell_agg k=100 emphasis
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "analysis_outputs", "r4_p1_trade_enriched.csv")
OUT_CSV = os.path.join(HERE, "analysis_outputs", "r4_p1_mark55_extract_by_day.csv")
OUT_JSON = os.path.join(HERE, "analysis_outputs", "r4_p1_mark55_extract_summary.json")


def stat(s: pd.Series) -> dict:
    x = s.dropna()
    n = int(len(x))
    if n < 2:
        return {"n": n, "mean": float("nan"), "std": float("nan"), "t": float("nan")}
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    t = m / (sd / np.sqrt(n)) if sd > 1e-12 else float("nan")
    return {"n": n, "mean": m, "std": sd, "t": t}


def main() -> None:
    df = pd.read_csv(SRC)
    ex = df[df["symbol"] == "VELVETFRUIT_EXTRACT"].copy()
    buy_side = ex[(ex["buyer"] == "Mark 55") & (ex["agg"] == "buy_agg")]
    sell_side = ex[(ex["seller"] == "Mark 55") & (ex["agg"] == "sell_agg")]

    rows = []
    for label, g in [("buy_agg", buy_side), ("sell_agg", sell_side)]:
        for k in (5, 20, 100):
            col = f"dm_self_k{k}"
            rows.append({"slice": label, "day": -1, "horizon_k": k, **stat(g[col])})
        for d, gd in g.groupby("day"):
            for k in (5, 20, 100):
                col = f"dm_self_k{k}"
                rows.append({"slice": label, "day": int(d), "horizon_k": k, **stat(gd[col])})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    summary = {
        "buy_agg_pooled": {f"k{k}": stat(buy_side[f"dm_self_k{k}"]) for k in (5, 20, 100)},
        "sell_agg_pooled": {f"k{k}": stat(sell_side[f"dm_self_k{k}"]) for k in (5, 20, 100)},
        "sell_agg_k100_by_day": {
            int(d): stat(g["dm_self_k100"]) for d, g in sell_side.groupby("day")
        },
        "n_buy_agg": int(len(buy_side)),
        "n_sell_agg": int(len(sell_side)),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
