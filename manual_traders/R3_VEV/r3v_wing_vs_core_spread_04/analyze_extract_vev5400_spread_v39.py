#!/usr/bin/env python3
"""Join extract book width to VEV_5400 book width by (day, timestamp) on Round 3 tapes."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

U = "VELVETFRUIT_EXTRACT"
V = "VEV_5400"


def spread_int(row: dict[str, str]) -> int | None:
    bps = []
    aps = []
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}", "").strip()
        ap = row.get(f"ask_price_{i}", "").strip()
        if bp:
            bps.append(int(bp))
        if ap:
            aps.append(int(ap))
    if not bps or not aps:
        return None
    return int(min(aps) - max(bps))


def main() -> None:
    root = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"
    pairs: list[tuple[int, int, int]] = []  # (u_sp, v_sp)
    u_by_slot: dict[tuple[int, int], int] = {}
    for day in (0, 1, 2):
        p = root / f"prices_round_3_day_{day}.csv"
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                try:
                    d = int(row["day"])
                    ts = int(row["timestamp"])
                except (KeyError, ValueError):
                    continue
                pr = row.get("product", "")
                sp = spread_int(row)
                if sp is None:
                    continue
                slot = (d, ts)
                if pr == U:
                    u_by_slot[slot] = sp
                elif pr == V and slot in u_by_slot:
                    pairs.append((u_by_slot[slot], sp))

    us = [a for a, _ in pairs]
    vs = [b for _, b in pairs]
    us_sorted = sorted(us)
    q75 = us_sorted[int(0.75 * (len(us_sorted) - 1))] if us_sorted else 0

    hi = [(u, v) for u, v in pairs if u >= q75]
    lo = [(u, v) for u, v in pairs if u < q75]

    def mean(xs: list[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    out = {
        "n_joined_rows": len(pairs),
        "extract_spread_mean": mean(us),
        "extract_spread_p50": us_sorted[len(us_sorted) // 2] if us_sorted else None,
        "extract_spread_p75": q75,
        "extract_spread_p90": us_sorted[int(0.90 * (len(us_sorted) - 1))] if len(us_sorted) > 1 else None,
        "vev5400_spread_mean_all": mean(vs),
        "vev5400_spread_mean_when_extract_ge_p75": mean([v for u, v in hi]),
        "vev5400_spread_mean_when_extract_lt_p75": mean([v for u, v in lo]),
        "n_hi": len(hi),
        "n_lo": len(lo),
    }
    outp = Path(__file__).resolve().parent / "extract_vev5400_spread_join_v39.json"
    outp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(outp.read_text())


if __name__ == "__main__":
    main()
