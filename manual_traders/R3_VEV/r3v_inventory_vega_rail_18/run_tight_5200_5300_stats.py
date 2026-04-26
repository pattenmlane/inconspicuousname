#!/usr/bin/env python3
"""BBO spread stats for VEV_5200 and VEV_5300; joint 'tight' = both ask-bid <= TH at same tick."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = (
    REPO
    / "manual_traders/R3_VEV/r3v_inventory_vega_rail_18"
    / "analysis_outputs"
    / "r3_tight_5200_5300_gate_by_day.json"
)
TH = 2
GATE = ("VEV_5200", "VEV_5300")


def main() -> None:
    by: dict = {}
    for day in (0, 1, 2):
        by_ts: dict[int, dict] = defaultdict(dict)
        with (DATA / f"prices_round_3_day_{day}.csv").open() as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                by_ts[int(row["timestamp"])][row["product"]] = row
        n = 0
        both = 0
        s52: list[int] = []
        s53: list[int] = []
        for ts in sorted(by_ts):
            d = by_ts[ts]
            ok = True
            sp: dict[str, int] = {}
            for p in GATE:
                rr = d.get(p)
                if not rr or not rr.get("bid_price_1") or not rr.get("ask_price_1"):
                    ok = False
                    break
                spr = int(rr["ask_price_1"]) - int(rr["bid_price_1"])
                sp[p] = spr
            if not ok:
                continue
            n += 1
            s52.append(sp["VEV_5200"])
            s53.append(sp["VEV_5300"])
            if sp["VEV_5200"] <= TH and sp["VEV_5300"] <= TH:
                both += 1
        import statistics

        by[str(day)] = {
            "timestamps_both_books": n,
            "share_joint_tight": both / n if n else 0.0,
            "mean_s5200": float(statistics.mean(s52)) if s52 else None,
            "mean_s5300": float(statistics.mean(s53)) if s53 else None,
            "p90_s5200": float(sorted(s52)[int(0.9 * (len(s52) - 1))]) if len(s52) > 1 else s52[0] if s52 else None,
            "p90_s5300": float(sorted(s53)[int(0.9 * (len(s53) - 1))]) if len(s53) > 1 else s53[0] if s53 else None,
        }
    out = {
        "definition": f"At each timestamp, both {GATE[0]} and {GATE[1]} have bid/ask; joint_tight = spr5200<={TH} and spr5300<={TH}.",
        "th": TH,
        "by_csv_day": by,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
