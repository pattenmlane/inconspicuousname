#!/usr/bin/env python3
"""Count Round 4 tape trades where Mark 67 is **buyer** on VELVETFRUIT_EXTRACT (per tape day file)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "r4_mark67_extract_buy_counts.json"

SYM = "VELVETFRUIT_EXTRACT"
MARK = "Mark 67"


def main() -> None:
    rows: list[dict] = []
    for p in sorted(DATA.glob("trades_round_4_day_*.csv")):
        day = int(p.stem.split("_")[-1].replace("day", "")) if "day" in p.stem else 0
        n = 0
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if row.get("symbol") != SYM:
                    continue
                if row.get("buyer") == MARK:
                    n += 1
        rows.append({"tape_file": str(p.relative_to(REPO)), "day_index": day, "mark67_buy_prints": n})
    total = sum(x["mark67_buy_prints"] for x in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    import json

    obj = {"product": SYM, "mark_buyer": MARK, "total_prints": total, "by_day": rows}
    OUT.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(json.dumps(obj, indent=2))


if __name__ == "__main__":
    main()
