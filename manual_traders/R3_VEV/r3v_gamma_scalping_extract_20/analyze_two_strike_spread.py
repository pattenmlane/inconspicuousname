#!/usr/bin/env python3
"""
Level-1 bid-ask spread for the two VEV strikes nearest to extract mid, by tape day.

From prices_round_3_day_*.csv: spread = ask_price_1 - bid_price_1 (when both set).
Tie break by timestamp step 200 for speed.
Output: analysis_outputs/two_strike_spread_by_day.json
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_TAPES = _ROOT / "Prosperity4Data" / "ROUND_3"
_OUT = Path(__file__).resolve().parent / "analysis_outputs"
UNDER = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]


def two_nearest(S: float) -> tuple[int, int]:
    r = sorted(STRIKES, key=lambda k: abs(float(k) - S))
    a, b = r[0], r[1]
    return (a, b) if a < b else (b, a)


def spread1(row) -> int | None:
    bp, ap = row.get("bid_price_1", ""), row.get("ask_price_1", "")
    if bp == "" or ap == "":
        return None
    return int(ap) - int(bp)


def main() -> None:
    summary: dict = {"method": "step 200; spread = ask1 - bid1 for VEV; two nearest strikes to extract mid", "by_tape_day": {}}
    for d in (0, 1, 2):
        path = _TAPES / f"prices_round_3_day_{d}.csv"
        by_ts: dict[int, dict[str, dict]] = {}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = int(row["timestamp"])
                if ts not in by_ts:
                    by_ts[ts] = {}
                by_ts[ts][row["product"]] = row
        sp_all: list[int] = []
        for ts in sorted(by_ts.keys())[::200]:
            rows = by_ts[ts]
            if UNDER not in rows:
                continue
            try:
                S = float(rows[UNDER]["mid_price"])
            except (KeyError, ValueError):
                continue
            k1, k2 = two_nearest(S)
            for k in (k1, k2):
                sym = f"VEV_{k}"
                if sym not in rows:
                    continue
                s = spread1(rows[sym])
                if s is not None:
                    sp_all.append(s)
        if sp_all:
            n = len(sp_all)
            summary["by_tape_day"][str(d)] = {
                "n": n,
                "mean_spread": sum(sp_all) / n,
            }
    _OUT.mkdir(parents=True, exist_ok=True)
    p = _OUT / "two_strike_spread_by_day.json"
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(p.read_text())


if __name__ == "__main__":
    main()
