#!/usr/bin/env python3
"""
Phase 2 cross-instrument slice: HYDROGEL_PACK Mark 38 buys from Mark 14 (tape duopoly).

Uses r4_p1_trade_enriched.csv + sonic_tight merge from r4_p3_trade_enriched_with_gate.csv.

Outputs:
- r4_p2c_hydro_m38_m14_summary.json — pooled / by day / by gate × day for dm_ex_k{5,20,100}
- r4_p2c_hydro_m38_m14_by_gate.csv — tight vs loose counts and means
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "analysis_outputs")
ENR = os.path.join(OUT, "r4_p1_trade_enriched.csv")
GATE = os.path.join(OUT, "r4_p3_trade_enriched_with_gate.csv")


def stat(s: pd.Series) -> dict:
    x = s.dropna()
    n = int(len(x))
    if n < 2:
        return {"n": n, "mean": float("nan"), "t": float("nan")}
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    t = m / (sd / np.sqrt(n)) if sd > 1e-12 else float("nan")
    return {"n": n, "mean": m, "t": t}


def main() -> None:
    df = pd.read_csv(ENR)
    h = df[(df["symbol"] == "HYDROGEL_PACK") & (df["buyer"] == "Mark 38") & (df["seller"] == "Mark 14")].copy()
    if os.path.isfile(GATE):
        g = pd.read_csv(GATE)[["day", "timestamp", "sonic_tight"]].drop_duplicates()
        h = h.merge(g, on=["day", "timestamp"], how="left")
        h["sonic_tight"] = h["sonic_tight"].fillna(False)
    else:
        h["sonic_tight"] = False

    summary: dict = {"slice": "HYDROGEL_PACK buyer Mark38 seller Mark14", "n": int(len(h))}
    for k in (5, 20, 100):
        col = f"dm_ex_k{k}"
        summary[f"pooled_{col}"] = stat(h[col])

    by_day = {}
    for d, g in h.groupby("day"):
        by_day[str(int(d))] = {f"k{k}": stat(g[f"dm_ex_k{k}"]) for k in (5, 20, 100)}
    summary["by_day"] = by_day

    gate_rows = []
    for tight, lab in [(True, "tight"), (False, "loose")]:
        sub = h[h["sonic_tight"] == tight]
        row = {"gate": lab, "n": int(len(sub))}
        for k in (5, 20, 100):
            col = f"dm_ex_k{k}"
            row[col] = stat(sub[col])["mean"]
            row[f"{col}_n"] = stat(sub[col])["n"]
        gate_rows.append(row)
    summary["by_gate"] = gate_rows

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(gate_rows).to_csv(os.path.join(OUT, "r4_p2c_hydro_m38_m14_by_gate.csv"), index=False)
    with open(os.path.join(OUT, "r4_p2c_hydro_m38_m14_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
