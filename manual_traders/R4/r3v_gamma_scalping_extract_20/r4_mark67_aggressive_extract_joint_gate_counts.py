#!/usr/bin/env python3
"""
Count **Mark 67** **aggressive** **VELVETFRUIT_EXTRACT** buys (price >= L1 ask)
per tape day, split by **Sonic joint tight** (5200+5300 spread<=2) at timestamp.

Output: analysis_outputs/r4_mark67_aggr_extract_joint_gate_counts_by_day.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = (1, 2, 3)
EXTRACT = "VELVETFRUIT_EXTRACT"
SURFACE = ("VEV_5200", "VEV_5300")
SPREAD_TH = 2
MARK67 = "Mark 67"


def joint_ts_for_day(day: int) -> set[int]:
    px = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    sub = px[px["product"].isin(SURFACE)].copy()
    sub["spr"] = pd.to_numeric(sub["ask_price_1"], errors="coerce") - pd.to_numeric(
        sub["bid_price_1"], errors="coerce"
    )
    pvt = sub.pivot_table(index="timestamp", columns="product", values="spr", aggfunc="first")
    pvt = pvt.dropna()
    jt = (pvt["VEV_5200"] <= SPREAD_TH) & (pvt["VEV_5300"] <= SPREAD_TH)
    return set(jt[jt].index.astype(int))


def main() -> None:
    rows = []
    for d in DAYS:
        joint = joint_ts_for_day(d)
        px = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        ex = px[px["product"] == EXTRACT][["timestamp", "bid_price_1", "ask_price_1"]].copy()
        ex["ts"] = ex["timestamp"].astype(int)
        ex["ask"] = pd.to_numeric(ex["ask_price_1"], errors="coerce")
        g = ex.groupby("ts", as_index=True).first()
        ask_map = {int(t): float(r["ask"]) for t, r in g.iterrows()}

        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr = tr[(tr["symbol"] == EXTRACT) & (tr["buyer"].fillna("").astype(str) == MARK67)]
        n_tight = n_loose = 0
        for _, r in tr.iterrows():
            t = int(r["timestamp"])
            if t not in ask_map:
                continue
            pr = float(r["price"])
            if pr < ask_map[t]:
                continue
            if t in joint:
                n_tight += 1
            else:
                n_loose += 1
        rows.append(
            {
                "day": d,
                "n_m67_aggr_joint_tight": n_tight,
                "n_m67_aggr_not_joint_tight": n_loose,
                "n_m67_aggr_total": n_tight + n_loose,
                "share_tight_given_aggr": round(n_tight / max(1, n_tight + n_loose), 6),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_mark67_aggr_extract_joint_gate_counts_by_day.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
