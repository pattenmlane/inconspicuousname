#!/usr/bin/env python3
"""
Phase 1 bullet 3 extension: directed buyer→seller edge counts; for top-N pairs by count,
report reverse edge (B→A) count, notional, and reciprocity = min(fwd, rev)/max.
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
TOP_N = 30


def main() -> None:
    edge_count: Counter[tuple[str, str]] = Counter()
    edge_notional: defaultdict[tuple[str, str], float] = defaultdict(float)
    for d in DAYS:
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                b = (r.get("buyer") or "").strip()
                s = (r.get("seller") or "").strip()
                if not b or not s:
                    continue
                edge_count[(b, s)] += 1
                try:
                    edge_notional[(b, s)] += abs(float(r["price"]) * float(r["quantity"]))
                except (KeyError, ValueError):
                    pass

    top = edge_count.most_common(TOP_N)
    rows = []
    for (a, b), c in top:
        rev = (b, a)
        rc = edge_count[rev]
        ntot = c + rc
        recip = min(c, rc) / max(c, rc) if max(c, rc) > 0 else None
        rows.append(
            {
                "forward": f"{a}->{b}",
                "n_forward": c,
                "n_reverse": rc,
                "n_total_undir": ntot,
                "reciprocity_min_max": recip,
                "notional_forward": round(edge_notional[(a, b)], 1),
                "notional_reverse": round(edge_notional[rev], 1),
            }
        )

    out = {"top_directed_edges_with_reciprocity": rows}
    pth = OUT / "r4_graph_reciprocity_top_edges.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
