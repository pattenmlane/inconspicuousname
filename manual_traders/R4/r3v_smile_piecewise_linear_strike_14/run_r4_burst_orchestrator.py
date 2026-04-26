#!/usr/bin/env python3
"""
Phase 1 bullet 4 extension: burst = >=3 trades same (day, timestamp). For each burst,
count buyer/seller names across all symbols; rank orchestrator candidates.

Outputs: summary counts, top buyer/seller roles across bursts, Mark01/M22 presence rates.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)


def load_trades_by_dt(day: int) -> dict[tuple[int, int], list[dict]]:
    by_dt: dict[tuple[int, int], list[dict]] = defaultdict(list)
    path = DATA / f"trades_round_4_day_{day}.csv"
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            ts = int(r["timestamp"])
            by_dt[(day, ts)].append(
                {
                    "buyer": (r.get("buyer") or "").strip(),
                    "seller": (r.get("seller") or "").strip(),
                    "sym": r["symbol"],
                }
            )
    return by_dt


def main() -> None:
    bursts: list[tuple[int, int, list[dict]]] = []
    for d in DAYS:
        by_dt = load_trades_by_dt(d)
        for (day, ts), rows in by_dt.items():
            if len(rows) >= 3:
                bursts.append((day, ts, rows))

    buyer_top1 = Counter()
    seller_top1 = Counter()
    mark01_buyer_burst = 0
    mark22_seller_burst = 0
    m01_m22_same_burst = 0

    for day, ts, rows in bursts:
        bc = Counter()
        sc = Counter()
        for r in rows:
            if r["buyer"]:
                bc[r["buyer"]] += 1
            if r["seller"]:
                sc[r["seller"]] += 1
        tb = bc.most_common(1)
        ts_ = sc.most_common(1)
        if tb:
            buyer_top1[tb[0][0]] += 1
        if ts_:
            seller_top1[ts_[0][0]] += 1
        buyers = {r["buyer"] for r in rows if r["buyer"]}
        sellers = {r["seller"] for r in rows if r["seller"]}
        if "Mark 01" in buyers:
            mark01_buyer_burst += 1
        if "Mark 22" in sellers:
            mark22_seller_burst += 1
        if "Mark 01" in buyers and "Mark 22" in sellers:
            m01_m22_same_burst += 1

    out = {
        "n_burst_timestamps": len(bursts),
        "frac_burst_with_mark01_as_any_buyer": mark01_buyer_burst / len(bursts) if bursts else None,
        "frac_burst_with_mark22_as_any_seller": mark22_seller_burst / len(bursts) if bursts else None,
        "frac_burst_with_mark01_buyer_and_mark22_seller": m01_m22_same_burst / len(bursts) if bursts else None,
        "top_mode_buyer_across_bursts": [{"name": n, "n_top1": c} for n, c in buyer_top1.most_common(12)],
        "top_mode_seller_across_bursts": [{"name": n, "n_top1": c} for n, c in seller_top1.most_common(12)],
    }
    pth = OUT / "r4_burst_orchestrator_modes.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
