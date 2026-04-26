#!/usr/bin/env python3
"""Forward VEV_5300 mid moves after Mark67→Mark22 extract + joint tight at print (15 events)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
SIG = Path(__file__).resolve().parent / "outputs" / "phase3" / "signals_mark67_to_mark22_extract_joint_tight_at_print.json"
OUT = Path(__file__).resolve().parent / "outputs" / "phase3" / "vev5300_fwd_after_6722_tight_signals.csv"


def mid_series(day: int, product: str) -> pd.Series:
    px = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    sub = px[px["product"] == product].drop_duplicates("timestamp").sort_values("timestamp")
    return sub.set_index("timestamp")["mid_price"].astype(float)


def fwd(s: pd.Series, ts: int, k: int) -> float:
    idx = s.index.to_numpy()
    pos = np.searchsorted(idx, ts)
    if pos >= len(idx) or int(idx[pos]) != ts:
        return float("nan")
    j = pos + k
    if j >= len(idx):
        return float("nan")
    a, b = float(s.iloc[pos]), float(s.iloc[j])
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return b - a


def main() -> None:
    sig = json.loads(SIG.read_text(encoding="utf-8"))
    s5300_by_day = {d: mid_series(d, "VEV_5300") for d in (1, 2, 3)}
    rows = []
    for tape_day, ts in sig:
        s = s5300_by_day[int(tape_day)]
        for k in (5, 20, 100):
            rows.append(
                {
                    "tape_day": int(tape_day),
                    "timestamp": int(ts),
                    "k": k,
                    "fwd_vev5300": fwd(s, int(ts), k),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    piv = df.pivot_table(index=["tape_day", "k"], values="fwd_vev5300", aggfunc=["mean", "median", "count"])
    piv.to_csv(OUT.with_name("vev5300_fwd_after_6722_summary_by_day_k.csv"))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
