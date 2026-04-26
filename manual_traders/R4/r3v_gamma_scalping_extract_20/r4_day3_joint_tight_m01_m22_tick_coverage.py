#!/usr/bin/env python3
"""
Day 3 only: share of **joint-tight** (5200+5300 spread≤2) timestamps where the
tape has a **Mark 01 → Mark 22** print on **VEV_5200** or **VEV_5300**.

Motivation: **v8** turns off *all* surface MM on day 3; **v10** only skips MM on
ticks with this print on day 3 — this CSV quantifies how often those differ.

Output: analysis_outputs/r4_day3_joint_tight_m01_m22_coverage.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAY = 3
SPREAD_TH = 2
SURFACE = ("VEV_5200", "VEV_5300")
MARK01, MARK22 = "Mark 01", "Mark 22"


def main() -> None:
    px = pd.read_csv(DATA / f"prices_round_4_day_{DAY}.csv", sep=";")
    px["spread"] = pd.to_numeric(px["ask_price_1"], errors="coerce") - pd.to_numeric(
        px["bid_price_1"], errors="coerce"
    )
    spr = px[px["product"].isin(SURFACE)][["timestamp", "product", "spread"]]
    pvt = spr.pivot_table(index="timestamp", columns="product", values="spread", aggfunc="first")
    pvt = pvt.dropna(subset=list(SURFACE))
    joint_tight_ts = set(
        pvt[(pvt["VEV_5200"] <= SPREAD_TH) & (pvt["VEV_5300"] <= SPREAD_TH)].index.astype(int)
    )
    n_joint = len(joint_tight_ts)

    tr = pd.read_csv(DATA / f"trades_round_4_day_{DAY}.csv", sep=";")
    tr = tr[tr["symbol"].isin(SURFACE)]
    tr["buyer"] = tr["buyer"].fillna("").astype(str)
    tr["seller"] = tr["seller"].fillna("").astype(str)
    m01_m22 = tr[(tr["buyer"] == MARK01) & (tr["seller"] == MARK22)]
    ts_with_print = set(m01_m22["timestamp"].astype(int))

    both = joint_tight_ts & ts_with_print
    n_both = len(both)
    share = (n_both / n_joint) if n_joint else 0.0

    row = {
        "tape_day": DAY,
        "n_joint_tight_timestamps": n_joint,
        "n_joint_tight_with_m01_m22_surface_print": n_both,
        "share_joint_tight_with_print": round(share, 6),
    }
    pd.DataFrame([row]).to_csv(OUT / "r4_day3_joint_tight_m01_m22_coverage.csv", index=False)
    print(row)


if __name__ == "__main__":
    main()
