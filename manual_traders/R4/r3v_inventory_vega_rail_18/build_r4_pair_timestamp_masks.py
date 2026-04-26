"""Build JSON timestamp sets for high-signal counterparty pairs (tape Round 4)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_pair_ts_masks.json"

PAIRS = [
    ("Mark 67", "Mark 22", "VELVETFRUIT_EXTRACT", "m67_m22_extract"),
    ("Mark 67", "Mark 49", "VELVETFRUIT_EXTRACT", "m67_m49_extract"),
]


def main() -> None:
    out: dict[str, dict[str, list[int]]] = {k: {} for _, _, _, k in PAIRS}
    for day in (1, 2, 3):
        path = DATA / f"trades_round_4_day_{day}.csv"
        sets: dict[str, set[int]] = {k: set() for _, _, _, k in PAIRS}
        with path.open() as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = int(row["timestamp"])
                for b, s, sym, key in PAIRS:
                    if row.get("buyer") == b and row.get("seller") == s and row["symbol"] == sym:
                        sets[key].add(ts)
        for key, s in sets.items():
            out[key][str(day)] = sorted(s)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT, {k: {d: len(v) for d, v in out[k].items()} for k in out})


if __name__ == "__main__":
    main()
