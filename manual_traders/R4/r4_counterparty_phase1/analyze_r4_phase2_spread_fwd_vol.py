#!/usr/bin/env python3
"""Phase 2.2: VEV_5300 — spread vs absolute forward 20-tick mid change (vol proxy)."""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_phase2_spread_fwd5300.json"
DAYS = (1, 2, 3)


def series(day: int):
    tss, sprs, mids = [], [], []
    path = DATA / f"prices_round_4_day_{day}.csv"
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            if int(row["day"]) != day or row["product"] != "VEV_5300":
                continue
            ts = int(row["timestamp"])
            bid = float(row["bid_price_1"])
            ask = float(row["ask_price_1"])
            mid = float(row["mid_price"])
            tss.append(ts)
            sprs.append(ask - bid)
            mids.append(mid)
    return tss, sprs, mids


def fwd_abs(tss, mids, ts, k):
    i = bisect_right(tss, ts) - 1
    if i < 0:
        i = 0
    j = i + k
    if j >= len(mids):
        return None
    return abs(mids[j] - mids[i])


def main() -> None:
    buckets: dict[str, list[float]] = defaultdict(list)
    for day in DAYS:
        tss, sprs, mids = series(day)
        for ts, sp, _ in zip(tss, sprs, mids):
            fa = fwd_abs(tss, mids, ts, 20)
            if fa is None:
                continue
            if sp <= 2:
                b = "spr_le2"
            elif sp <= 6:
                b = "spr_3_6"
            else:
                b = "spr_gt6"
            buckets[b].append(fa)
    out = {
        "mean_abs_fwd20_mid_by_spread_bucket": {
            k: round(sum(v) / len(v), 6) for k, v in buckets.items() if v
        },
        "n": {k: len(v) for k, v in buckets.items()},
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
