#!/usr/bin/env python3
"""
At each Mark14/M38 aggr_sell VEV_4000 print: attach **rolling** joint-tight fraction (WIN=400,
same as v7/v15) from aligned 5200/5300 panel.

Output: r4_p17_m14_m38_vev4000_print_roll_fwd20.csv
Summary printed: day-3 mean fwd20 when roll>thr vs roll<=thr for thr in (0.38, 0.42, 0.45).
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
P1 = BASE / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
PAN = BASE / "outputs_r4_phase3" / "r4_p3_joint_gate_panel_by_timestamp.csv"
OUT = BASE / "outputs_r4_phase3" / "r4_p17_m14_m38_vev4000_print_roll_fwd20.csv"
WIN = 400


def add_roll(pan: pd.DataFrame) -> pd.Series:
    pan = pan.sort_values("timestamp").reset_index(drop=True)
    q: deque[int] = deque(maxlen=WIN)
    rates = []
    for t in pan["tight"].astype(bool):
        q.append(1 if t else 0)
        rates.append(sum(q) / len(q))
    return pd.Series(rates, index=pan.index)


def main() -> None:
    tr = pd.read_csv(P1)
    sub = tr[
        (tr["symbol"] == "VEV_4000")
        & (tr["buyer"] == "Mark 14")
        & (tr["seller"] == "Mark 38")
        & (tr["aggressor_bucket"] == "aggr_sell")
    ][["day", "timestamp", "fwd_mid_k20"]].copy()
    sub["fwd20"] = pd.to_numeric(sub["fwd_mid_k20"], errors="coerce")

    pan_all = pd.read_csv(PAN)
    rolls = []
    for d in sorted(sub["day"].unique()):
        pan = pan_all[pan_all["day"] == d].sort_values("timestamp").reset_index(drop=True)
        pan["roll_tight_rate"] = add_roll(pan)
        m = dict(zip(pan["timestamp"].astype(int), pan["roll_tight_rate"].astype(float)))
        for _, r in sub[sub["day"] == d].iterrows():
            ts = int(r["timestamp"])
            rolls.append(m.get(ts, float("nan")))
    sub["roll_tight_rate"] = rolls
    sub.to_csv(OUT, index=False)

    d3 = sub[sub["day"] == 3].dropna(subset=["fwd20", "roll_tight_rate"])
    print("day3 n", len(d3), "roll mean", d3["roll_tight_rate"].mean(), "p90", d3["roll_tight_rate"].quantile(0.9))
    for thr in [0.38, 0.40, 0.42, 0.45]:
        hi = d3[d3["roll_tight_rate"] > thr]["fwd20"]
        lo = d3[d3["roll_tight_rate"] <= thr]["fwd20"]
        if len(hi) >= 3 and len(lo) >= 3:
            print(
                f"thr={thr}: hi n={len(hi)} mean={hi.mean():.3f} | lo n={len(lo)} mean={lo.mean():.3f}"
            )
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
