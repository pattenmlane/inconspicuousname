#!/usr/bin/env python3
"""Phase 2.1: Mark 01→Mark 22 multi-symbol bursts — forward VEV_5300 mid vs controls.

Burst: (day, ts) with >=4 trade rows and at least one row buyer Mark 01 seller Mark 22.
Forward: K rows ahead in prices_round_4 for VEV_5300 (same day).
Control: random (day, ts) with same burst size distribution from non-burst timestamps (seeded).
"""
from __future__ import annotations

import csv
import json
import random
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_phase2_burst_m01_m22.json"
DAYS = (1, 2, 3)
KS = (5, 20, 50)
SEED = 42
N_CTRL = 400


def load_prices_5300() -> dict[int, tuple[list[int], list[float]]]:
    out: dict[int, tuple[list[int], list[float]]] = {}
    for day in DAYS:
        path = DATA / f"prices_round_4_day_{day}.csv"
        mp: dict[int, float] = {}
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day or row["product"] != "VEV_5300":
                    continue
                mp[int(row["timestamp"])] = float(row["mid_price"])
        tss = sorted(mp)
        mids = [mp[t] for t in tss]
        out[day] = (tss, mids)
    return out


def fwd(tss: list[int], mids: list[float], ts: int, k: int) -> float | None:
    i = bisect_right(tss, ts) - 1
    if i < 0:
        i = 0
    j = i + k
    if j >= len(mids):
        return None
    return mids[j] - mids[i]


def main() -> None:
    random.seed(SEED)
    p5300 = load_prices_5300()
    burst_keys: list[tuple[int, int]] = []
    burst_size: dict[tuple[int, int], int] = {}
    m01_m22_at_ts: dict[tuple[int, int], bool] = defaultdict(bool)

    for day in DAYS:
        tp = DATA / f"trades_round_4_day_{day}.csv"
        by_ts: dict[int, list[dict]] = defaultdict(list)
        with open(tp, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                ts = int(row["timestamp"])
                by_ts[ts].append(row)
        for ts, rows in by_ts.items():
            n = len(rows)
            burst_size[(day, ts)] = n
            if n >= 4 and any(
                str(x["buyer"]) == "Mark 01" and str(x["seller"]) == "Mark 22"
                for x in rows
            ):
                burst_keys.append((day, ts))
                m01_m22_at_ts[(day, ts)] = True

    non_burst = [
        (d, t)
        for (d, t) in burst_size
        if (d, t) not in m01_m22_at_ts and burst_size[(d, t)] == 1
    ]

    def collect(keys: list[tuple[int, int]], label: str) -> dict:
        rows_out = {f"fwd_{k}": [] for k in KS}
        for day, ts in keys:
            tss, mids = p5300[day]
            for k in KS:
                dx = fwd(tss, mids, ts, k)
                if dx is not None:
                    rows_out[f"fwd_{k}"].append(dx)
        return {
            "label": label,
            "n": len(keys),
            **{
                f"mean_fwd_{k}": round(sum(v) / len(v), 6) if v else None
                for k, v in [(k, rows_out[f"fwd_{k}"]) for k in KS]
            },
        }

    ctrl = random.sample(non_burst, min(N_CTRL, len(non_burst)))
    out = {
        "n_m01_m22_burst_ge4": len(burst_keys),
        "burst_event": collect(burst_keys, "m01_m22_burst_ge4"),
        "control_singleton": collect(ctrl, "singleton_control_sample"),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
