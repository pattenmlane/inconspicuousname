#!/usr/bin/env python3
"""Share of timestamps (per tape day) where VEV_5200 and VEV_5300 are both L1-tight (spread≤2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = (1, 2, 3)
SURFACE = ("VEV_5200", "VEV_5300")
SPREAD_TH = 2


def main() -> None:
    rows = []
    for d in DAYS:
        px = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        ts_all = int(px["timestamp"].nunique())
        sub = px[px["product"].isin(SURFACE)][["timestamp", "product", "bid_price_1", "ask_price_1"]].copy()
        sub["spread"] = pd.to_numeric(sub["ask_price_1"], errors="coerce") - pd.to_numeric(
            sub["bid_price_1"], errors="coerce"
        )
        pvt = sub.pivot_table(index="timestamp", columns="product", values="spread", aggfunc="first")
        pvt = pvt.dropna(subset=list(SURFACE))
        n_joint = int(((pvt["VEV_5200"] <= SPREAD_TH) & (pvt["VEV_5300"] <= SPREAD_TH)).sum())
        rows.append(
            {
                "tape_day": d,
                "n_unique_timestamps_prices": ts_all,
                "n_joint_tight_timestamps": n_joint,
                "share_joint_tight": round(n_joint / max(1, ts_all), 6),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_joint_tight_time_share_by_day.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
