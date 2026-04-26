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
OUT2 = Path(__file__).resolve().parent / "outputs" / "phase3" / "extract_vev5300_joint_fwd_after_6722_tight.csv"


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
    sext_by_day = {d: mid_series(d, "VELVETFRUIT_EXTRACT") for d in (1, 2, 3)}
    rows = []
    rows2 = []
    for tape_day, ts in sig:
        d = int(tape_day)
        s = s5300_by_day[d]
        se = sext_by_day[d]
        for k in (5, 20, 100):
            f5300 = fwd(s, int(ts), k)
            fext = fwd(se, int(ts), k)
            rows.append(
                {
                    "tape_day": d,
                    "timestamp": int(ts),
                    "k": k,
                    "fwd_vev5300": f5300,
                }
            )
            rows2.append(
                {
                    "tape_day": d,
                    "timestamp": int(ts),
                    "k": k,
                    "fwd_extract": fext,
                    "fwd_vev5300": f5300,
                    "sum_fwd": float(fext + f5300)
                    if (fext == fext and f5300 == f5300)
                    else float("nan"),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    piv = df.pivot_table(index=["tape_day", "k"], values="fwd_vev5300", aggfunc=["mean", "median", "count"])
    piv.to_csv(OUT.with_name("vev5300_fwd_after_6722_summary_by_day_k.csv"))

    jdf = pd.DataFrame(rows2).dropna(subset=["fwd_extract", "fwd_vev5300"])
    jdf.to_csv(OUT2, index=False)
    corr_rows = []
    for k in (5, 20, 100):
        sub = jdf[jdf["k"] == k]
        if len(sub) >= 3:
            r = float(sub["fwd_extract"].corr(sub["fwd_vev5300"]))
        else:
            r = float("nan")
        corr_rows.append({"k": k, "corr_fwd_extract_fwd_5300": r, "n": len(sub)})
    pd.DataFrame(corr_rows).to_csv(OUT2.with_name("extract_vev5300_fwd_corr_by_k.csv"), index=False)

    print("Wrote", OUT, "and", OUT2)


if __name__ == "__main__":
    main()
