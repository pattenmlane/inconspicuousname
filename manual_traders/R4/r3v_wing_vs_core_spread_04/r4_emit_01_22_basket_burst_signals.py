#!/usr/bin/env python3
"""
Emit merged-timestamp fire set for Mark 01 → Mark 22 **basket bursts** (Phase 2 definition).

Same rule as r4_phase2_analysis.py: group trades by (tape_day, timestamp) where
buyer==Mark 01 and seller==Mark 22; keep timestamps with ≥4 rows in the group.
Causality-safe fire time: local trade timestamp + LAG (100), merged into global clock
as (tape_day - 1) * 1_000_000 + local_ts + LAG — matches trader_v1 Mark67 convention.

Output: outputs/phase2/signals_01_22_basket_burst_fire.json
  { "lag": 100, "min_prints": 4, "n_bursts": N, "fires_merged_ts": [...] }
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
LAG = 100
MIN_PRINTS = 4


def main() -> None:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        x = pd.read_csv(p, sep=";")
        x["tape_day"] = d
        frames.append(x)
    m = pd.concat(frames, ignore_index=True)
    sub = m[(m["buyer"] == "Mark 01") & (m["seller"] == "Mark 22")]
    g = sub.groupby(["tape_day", "timestamp"]).size().reset_index(name="n_prints")
    burst = g[g["n_prints"] >= MIN_PRINTS]
    fires: list[int] = []
    for _, r in burst.iterrows():
        d, t = int(r["tape_day"]), int(r["timestamp"])
        merged = (d - 1) * 1_000_000 + t + LAG
        fires.append(merged)
    fires = sorted(set(fires))
    payload = {
        "lag": LAG,
        "min_prints": MIN_PRINTS,
        "n_burst_timestamps": int(len(burst)),
        "n_unique_fire_merged": len(fires),
        "fires_merged_ts": fires,
    }
    out_path = OUT / "signals_01_22_basket_burst_fire.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path, "written", len(fires), "fire timestamps")


if __name__ == "__main__":
    main()
