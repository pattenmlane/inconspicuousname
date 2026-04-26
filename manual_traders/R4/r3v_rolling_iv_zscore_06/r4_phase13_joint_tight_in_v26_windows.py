#!/usr/bin/env python3
"""Tape: fraction of **price-grid** rows in each v26 Mark67 window where Sonic joint gate is tight."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SIG = Path(__file__).resolve().parent / "outputs" / "r4_v26_signals.json"
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
TH = 2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = json.loads(SIG.read_text())
    triggers = sorted(int(x) for x in obj.get("mark67_extract_buy_aggr_filtered_merged_ts", []))
    W = int(obj.get("window_ts", 50_000))
    cum = {int(k): int(v) for k, v in obj.get("day_cum_offset", {}).items()}

    px = []
    for d in (1, 2, 3):
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            px.append(df)
    px = pd.concat(px, ignore_index=True)
    sp = px[px["product"].isin(["VEV_5200", "VEV_5300"])].copy()
    sp["spread"] = sp["ask_price_1"] - sp["bid_price_1"]
    sp52 = sp[sp["product"] == "VEV_5200"][["day", "timestamp", "spread"]].rename(columns={"spread": "s52"})
    sp53 = sp[sp["product"] == "VEV_5300"][["day", "timestamp", "spread"]].rename(columns={"spread": "s53"})
    jt = sp52.merge(sp53, on=["day", "timestamp"])
    jt["joint_tight"] = (jt["s52"] <= TH) & (jt["s53"] <= TH)
    jt["merged_ts"] = jt["day"].map(cum) + jt["timestamp"]
    jt = jt.sort_values("merged_ts")

    ts_all = jt["merged_ts"].to_numpy(dtype=np.int64)
    tight = jt["joint_tight"].to_numpy()

    rows = []
    for T in triggers:
        lo, hi = int(T), int(T + W)
        m = (ts_all >= lo) & (ts_all <= hi)
        n = int(m.sum())
        if n == 0:
            frac = float("nan")
        else:
            frac = float(tight[m].mean())
        rows.append({"trigger_merged_ts": T, "n_price_rows_in_window": n, "frac_joint_tight": frac})

    df = pd.DataFrame(rows)
    p_csv = OUT / "r4_p13_joint_tight_fraction_per_v26_window.csv"
    df.to_csv(p_csv, index=False)
    good = df["frac_joint_tight"].dropna()
    lines = [
        "Sonic joint gate (5200&5300 spread<=2) **time fraction** inside each v26 window (price rows only).\n",
        f"n_triggers={len(df)} W={W}\n",
        f"frac_joint_tight: mean={good.mean():.3f} median={good.median():.3f} min={good.min():.3f} max={good.max():.3f}\n",
    ]
    (OUT / "r4_p13_joint_tight_in_v26_windows.txt").write_text("".join(lines))
    print(f"Wrote {p_csv} and {OUT / 'r4_p13_joint_tight_in_v26_windows.txt'}")


if __name__ == "__main__":
    main()
