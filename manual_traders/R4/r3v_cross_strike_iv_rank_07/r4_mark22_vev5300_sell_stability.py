#!/usr/bin/env python3
"""
Day-stability for Phase-1 headline: Mark 22 aggressive sells on VEV_5300 (tape-only markouts).

Reads r4_p1_trade_enriched.csv (Phase 1 script output). Aggressive sell = agg == 'sell_agg'.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "analysis_outputs", "r4_p1_trade_enriched.csv")
OUT = os.path.join(HERE, "analysis_outputs", "r4_p1_mark22_vev5300_sell_agg_by_day.csv")


def _summ(x: pd.Series) -> dict:
    v = x.dropna()
    n = len(v)
    if n < 5:
        return {"n": n, "mean": np.nan, "t": np.nan, "pos_frac": np.nan}
    m = float(v.mean())
    s = float(v.std(ddof=1)) if n > 1 else np.nan
    t = m / (s / np.sqrt(n)) if s and s > 1e-12 and s == s else np.nan
    return {"n": n, "mean": m, "t": t, "pos_frac": float((v > 0).mean())}


def main() -> None:
    df = pd.read_csv(SRC)
    m = df[
        (df["symbol"] == "VEV_5300")
        & (df["seller"] == "Mark 22")
        & (df["agg"] == "sell_agg")
    ].copy()
    rows = []
    for k in (5, 20, 100):
        col = f"dm_self_k{k}"
        for day, g in m.groupby("day"):
            rows.append({"horizon_k": k, "day": int(day), **_summ(g[col])})
        rows.append({"horizon_k": k, "day": -1, **_summ(m[col])})  # pooled
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
