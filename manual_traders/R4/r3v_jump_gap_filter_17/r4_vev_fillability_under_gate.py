#!/usr/bin/env python3
"""Probe: when Sonic joint gate is on, are VEV_5200/5300 books often 1-tick wide (no room for
improving maker quotes that still cross the spread for the backtester's passive fill path)?

Writes: manual_traders/R4/r3v_jump_gap_filter_17/outputs/phase3/r4_vev_spread_under_joint_gate.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = (1, 2, 3)


def strip(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = df[df["product"] == product].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    return v.assign(spread=(ask - bid).astype(int))[["timestamp", "spread"]]


def main() -> None:
    rows = []
    for day in DAYS:
        pr = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        a = strip(pr, "VEV_5200").rename(columns={"spread": "s52"})
        b = strip(pr, "VEV_5300").rename(columns={"spread": "s53"})
        m = a.merge(b, on="timestamp", how="inner")
        m["joint_tight"] = (m["s52"] <= 2) & (m["s53"] <= 2)
        j = m[m["joint_tight"]]
        if len(j) == 0:
            continue
        for sym, col in [("VEV_5200", "s52"), ("VEV_5300", "s53")]:
            sp = j[col]
            rows.append(
                {
                    "day": day,
                    "symbol": sym,
                    "n_joint_ticks": len(j),
                    "frac_spread_1": float((sp == 1).mean()),
                    "frac_spread_2": float((sp == 2).mean()),
                    "mean_spread": float(sp.mean()),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "r4_vev_spread_under_joint_gate.csv", index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
