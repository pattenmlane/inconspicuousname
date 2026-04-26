#!/usr/bin/env python3
"""Distribution of VEV_5300 forward mids after tight+burst timestamps (Round 4 days 1-3).

Same gate/burst definition as r4_gate_burst_5300_fwd_2x2.json; report mean and
fraction positive for K in {5,10,20,50} on the tight+burst subset only.
"""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_tight_burst_5300_fwd_horizons.json"
DAYS = (1, 2, 3)
TH = 2.0
BURST_MIN = 4
KS = (5, 10, 20, 50)
S5200, S5300 = "VEV_5200", "VEV_5300"


def load_burst_flag():
    by: dict[tuple[int, int], list] = defaultdict(list)
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        with open(p, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                by[(d, int(row["timestamp"]))].append(row)
    out = {}
    for key, rows in by.items():
        if len(rows) < BURST_MIN:
            out[key] = False
            continue
        out[key] = any(
            str(x["buyer"]).strip() == "Mark 01" and str(x["seller"]).strip() == "Mark 22"
            for x in rows
        )
    return out


def fwd(tss, mids, ts, k):
    i = bisect_right(tss, ts) - 1
    if i < 0:
        i = 0
    j = i + k
    if j >= len(mids):
        return None
    return mids[j] - mids[i]


def main() -> None:
    burst_ok = load_burst_flag()
    by_k: dict[int, list[float]] = {k: [] for k in KS}

    for day in DAYS:
        sp52: dict[int, float] = {}
        sp53: dict[int, float] = {}
        tss53: list[int] = []
        mids53: list[float] = []
        path = DATA / f"prices_round_4_day_{day}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day:
                    continue
                sym = row["product"]
                ts = int(row["timestamp"])
                bid = int(float(row["bid_price_1"]))
                ask = int(float(row["ask_price_1"]))
                sp = ask - bid if ask > bid else 0
                if sym == S5200:
                    sp52[ts] = float(sp)
                elif sym == S5300:
                    sp53[ts] = float(sp)
                    tss53.append(ts)
                    mids53.append(float(row["mid_price"]))
        for ts in tss53:
            if ts not in sp52 or ts not in sp53:
                continue
            if sp52[ts] > TH or sp53[ts] > TH:
                continue
            if not burst_ok.get((day, ts), False):
                continue
            for k in KS:
                d = fwd(tss53, mids53, ts, k)
                if d is not None:
                    by_k[k].append(d)

    out = {"subset": "tight_and_burst", "TH": TH}
    for k in KS:
        xs = by_k[k]
        if not xs:
            out[f"K{k}"] = {"n": 0}
            continue
        pos = sum(1 for x in xs if x > 0) / len(xs)
        out[f"K{k}"] = {
            "n": len(xs),
            "mean": round(sum(xs) / len(xs), 6),
            "frac_positive": round(pos, 4),
        }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
