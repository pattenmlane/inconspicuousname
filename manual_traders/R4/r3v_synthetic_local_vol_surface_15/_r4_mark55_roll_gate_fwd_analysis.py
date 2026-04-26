#!/usr/bin/env python3
"""
Mark 55 aggressive-buy EXTRACT: forward mid K=20 vs **rolling joint-tight prevalence**
(WIN=400, buffer resets per historical day) at same timestamp.

Outputs tertile of roll_tight_rate within each day, then mean fwd_mid_k20 per tertile × day.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
P1 = Path(__file__).resolve().parent / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "outputs_r4_phase3" / "r4_p7_mark55_fwd20_by_roll_tertile.csv"
DAYS = [1, 2, 3]
TH = 2
WIN = 400


def roll_series_for_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    rows = []
    for sym in ["VEV_5200", "VEV_5300"]:
        sub = df[df["product"] == sym].drop_duplicates("timestamp").sort_values("timestamp")
        bid = pd.to_numeric(sub["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(sub["ask_price_1"], errors="coerce")
        sub = sub.assign(
            spread=(ask - bid).astype(float),
            sym=sym,
        )[["timestamp", "spread", "sym"]]
        rows.append(sub)
    x = pd.concat(rows, ignore_index=True)
    p5200 = x[x["sym"] == "VEV_5200"].rename(columns={"spread": "s5200"})[["timestamp", "s5200"]]
    p5300 = x[x["sym"] == "VEV_5300"].rename(columns={"spread": "s5300"})[["timestamp", "s5300"]]
    m = p5200.merge(p5300, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)

    q: deque[int] = deque(maxlen=WIN)
    rates = []
    for t in m["tight"]:
        q.append(1 if t else 0)
        rates.append(sum(q) / len(q))
    m["roll_tight_rate"] = rates
    m["day"] = day
    return m[["day", "timestamp", "roll_tight_rate"]]


def main() -> None:
    roll = pd.concat([roll_series_for_day(d) for d in DAYS], ignore_index=True)
    tr = pd.read_csv(P1)
    m55 = tr[
        (tr["symbol"] == "VELVETFRUIT_EXTRACT")
        & (tr["buyer"] == "Mark 55")
        & (tr["aggressor_bucket"] == "aggr_buy")
    ].merge(roll, on=["day", "timestamp"], how="inner")
    m55["fwd20"] = pd.to_numeric(m55["fwd_mid_k20"], errors="coerce")

    out_rows = []
    for d in DAYS:
        sub = m55[m55["day"] == d].dropna(subset=["roll_tight_rate", "fwd20"])
        if len(sub) < 30:
            continue
        try:
            sub = sub.copy()
            sub["tert"] = pd.qcut(
                sub["roll_tight_rate"].rank(method="first"),
                3,
                labels=["low_roll", "mid_roll", "high_roll"],
                duplicates="drop",
            )
        except Exception:
            continue
        for lab in ["low_roll", "mid_roll", "high_roll"]:
            s2 = sub[sub["tert"] == lab]["fwd20"]
            if len(s2) < 5:
                continue
            out_rows.append(
                {
                    "day": d,
                    "tert": lab,
                    "n": int(len(s2)),
                    "mean_fwd20": float(s2.mean()),
                    "median_fwd20": float(s2.median()),
                }
            )
    pd.DataFrame(out_rows).to_csv(OUT, index=False)
    print(pd.DataFrame(out_rows).to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
