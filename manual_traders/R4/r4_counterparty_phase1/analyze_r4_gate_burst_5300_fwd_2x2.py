#!/usr/bin/env python3
"""2x2: Sonic joint gate × M01→M22 burst vs VEV_5300 K-step forward mid (Round 4 days 1-3)."""
from __future__ import annotations

import csv
import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_gate_burst_5300_fwd_2x2.json"
DAYS = (1, 2, 3)
TH = 2.0
K = 20
BURST_MIN = 4
S5200, S5300 = "VEV_5200", "VEV_5300"


def load_burst_flag():
    by: dict[tuple[int, int], list] = defaultdict(list)
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        with open(p, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                by[(d, int(row["timestamp"]))].append(row)
    out: dict[tuple[int, int], bool] = {}
    for key, rows in by.items():
        if len(rows) < BURST_MIN:
            out[key] = False
            continue
        out[key] = any(
            str(x["buyer"]).strip() == "Mark 01" and str(x["seller"]).strip() == "Mark 22"
            for x in rows
        )
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
    burst_ok = load_burst_flag()
    cells: dict[str, list[float]] = defaultdict(list)

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
                bid = float(row["bid_price_1"])
                ask = float(row["ask_price_1"])
                sp = ask - bid if ask > bid else 0.0
                if sym == S5200:
                    sp52[ts] = sp
                elif sym == S5300:
                    sp53[ts] = sp
                    tss53.append(ts)
                    mids53.append(float(row["mid_price"]))
        if len(tss53) < K + 2:
            continue
        for ts in tss53:
            if ts not in sp52 or ts not in sp53:
                continue
            tight = sp52[ts] <= TH and sp53[ts] <= TH
            br = burst_ok.get((day, ts), False)
            fk = fwd(tss53, mids53, ts, K)
            if fk is None:
                continue
            key = f"tight={tight}_burst={br}"
            cells[key].append(fk)

    def stat(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": None}
        return {"n": len(xs), "mean": round(sum(xs) / len(xs), 6)}

    out = {
        "K": K,
        "TH": TH,
        "burst_min_rows": BURST_MIN,
        "cells": {k: stat(v) for k, v in sorted(cells.items())},
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
