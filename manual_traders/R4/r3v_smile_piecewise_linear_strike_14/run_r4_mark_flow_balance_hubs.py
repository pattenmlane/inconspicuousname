#!/usr/bin/env python3
"""
Phase 1 — participant clustering: per Mark name, signed flow (buyer minus seller notional
count), undirected graph degree, hub score (count of distinct counterparty names).

Excludes non-Mark rows. Days 1-3 pooled.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)


def is_mark(n: str) -> bool:
    n = n.strip()
    return n.startswith("Mark ")


def main() -> None:
    print_count: Counter[str] = Counter()
    neighbors: dict[str, set[str]] = defaultdict(set)
    as_buyer_not: dict[str, float] = defaultdict(float)
    as_sell_not: dict[str, float] = defaultdict(float)

    for d in DAYS:
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                b = (r.get("buyer") or "").strip()
                s = (r.get("seller") or "").strip()
                try:
                    ntl = abs(float(r["price"]) * float(r["quantity"]))
                except (KeyError, ValueError):
                    continue
                for name in (b, s):
                    if is_mark(name):
                        print_count[name] += 1
                if is_mark(b):
                    as_buyer_not[b] += ntl
                if is_mark(s):
                    as_sell_not[s] += ntl
                if is_mark(b) and is_mark(s):
                    neighbors[b].add(s)
                    neighbors[s].add(b)

    all_marks = set(as_buyer_not) | set(as_sell_not) | set(print_count) | set(neighbors)
    marks = sorted(n for n in all_marks if is_mark(n))
    rows = []
    for m in marks:
        gb = as_buyer_not.get(m, 0.0)
        gs = as_sell_not.get(m, 0.0)
        deg = len(neighbors.get(m, set()))
        rows.append(
            {
                "mark": m,
                "n_prints_touched": print_count.get(m, 0),
                "n_distinct_counterparties_mark_to_mark": deg,
                "signed_notional_buy_minus_sell": gb - gs,
                "gross_as_buyer": gb,
                "gross_as_seller": gs,
            }
        )
    rows.sort(key=lambda r: -r["n_prints_touched"])
    by_deg = sorted(rows, key=lambda r: -r["n_distinct_counterparties_mark_to_mark"])

    out = {
        "method": "Each trade: only Mark-Mark notional included in buy-minus-sell proxy; all trades count for participation and neighbor sets when both sides are Marks.",
        "top_by_prints": rows[:15],
        "top_by_distinct_counterparties": by_deg[:15],
        "pooled_median_net_signed": statistics.median(
            [r["signed_notional_buy_minus_sell"] for r in rows]
        )
        if rows
        else None,
    }
    pth = OUT / "r4_mark_flow_balance_hubs.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
