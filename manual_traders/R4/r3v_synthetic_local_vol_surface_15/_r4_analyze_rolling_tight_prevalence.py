#!/usr/bin/env python3
"""Rolling fraction of Sonic-tight ticks vs day (Round 4 prices); motivates rarity filter."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs_r4_phase3" / "r4_p6_rolling_tight_prevalence_by_day.csv"
DAYS = [1, 2, 3]
TH = 2
WIN = 400


def panel_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    rows = []
    for sym in ["VEV_5200", "VEV_5300"]:
        sub = df[df["product"] == sym].drop_duplicates("timestamp").sort_values("timestamp")
        bid = pd.to_numeric(sub["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(sub["ask_price_1"], errors="coerce")
        sub = sub.assign(spread=(ask - bid).astype(float))[["timestamp", "spread"]]
        sub["sym"] = sym
        rows.append(sub)
    x = pd.concat(rows, ignore_index=True)
    p5200 = x[x["sym"] == "VEV_5200"].rename(columns={"spread": "s5200"})[["timestamp", "s5200"]]
    p5300 = x[x["sym"] == "VEV_5300"].rename(columns={"spread": "s5300"})[["timestamp", "s5300"]]
    m = p5200.merge(p5300, on="timestamp", how="inner").sort_values("timestamp")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["day"] = day
    return m


def main() -> None:
    all_rows = []
    for d in DAYS:
        m = panel_day(d)
        q = deque(maxlen=WIN)
        roll = []
        for t in m["tight"]:
            q.append(1 if t else 0)
            roll.append(sum(q) / len(q))
        m["roll_tight_rate"] = roll
        all_rows.append(m)
    pan = pd.concat(all_rows, ignore_index=True)
    summ = pan.groupby("day")["roll_tight_rate"].agg(
        mean="mean", p50="median", p90=lambda s: float(np.quantile(s, 0.9))
    ).reset_index()
    summ.to_csv(OUT, index=False)
    print(summ.to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
