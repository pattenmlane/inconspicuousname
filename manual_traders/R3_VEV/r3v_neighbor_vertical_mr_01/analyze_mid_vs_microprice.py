#!/usr/bin/env python3
"""Compare simple mid vs L1 microprice for VEV_5000 on Round 3 tapes (book proxy)."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_mid_vs_microprice_vev5000.json"
SYM = "VEV_5000"


def load(path: Path):
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open() as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            by_ts[int(row["timestamp"])][row["product"]] = row
    return by_ts


def mid_and_micro(row: dict) -> tuple[float | None, float | None]:
    try:
        bb = int(row["bid_price_1"])
        ba = int(row["ask_price_1"])
        bv = int(row["bid_volume_1"])
        av = int(row["ask_volume_1"])
    except (KeyError, ValueError):
        return None, None
    mid = 0.5 * (bb + ba)
    if bv + av <= 0:
        return mid, mid
    micro = (bb * av + ba * bv) / float(bv + av)
    return mid, micro


def main():
    diffs: list[float] = []
    for d in (0, 1, 2):
        p = load(DATA / f"prices_round_3_day_{d}.csv")
        for ts in sorted(p.keys())[:3000:5]:
            row = p[ts].get(SYM)
            if row is None:
                continue
            mid, micro = mid_and_micro(row)
            if mid is None or micro is None:
                continue
            diffs.append(micro - mid)
    n = len(diffs)
    if n == 0:
        doc = {"n": 0, "error": "no samples"}
    else:
        m = sum(diffs) / n
        v = sum((x - m) ** 2 for x in diffs) / max(n - 1, 1)
        doc = {
            "symbol": SYM,
            "n": n,
            "mean_micro_minus_mid": m,
            "std": math.sqrt(v),
            "method": "L1 microprice (bid*askVol+ask*bidVol)/(bidVol+askVol); mid=(bid+ask)/2 from CSV.",
        }
    OUT.write_text(json.dumps(doc, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
