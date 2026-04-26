#!/usr/bin/env python3
"""
Round 3 tapes (day 0-2): per-VEV mean half-spread, beta (from tape deltas),
and opportunity proxy beta/mean_h for ranking vs simple |z| selection.
Input: Prosperity4Data/ROUND_3/prices_round_3_day_*.csv
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"
DAYS = (0, 1, 2)
SYMS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
UNDER = "VELVETFRUIT_EXTRACT"

# OLS betas from prior analysis (same tape); included for one JSON stop
BETA_TAPE = {
    "VEV_4000": 0.745,
    "VEV_4500": 0.662,
    "VEV_5000": 0.654,
    "VEV_5100": 0.577,
    "VEV_5200": 0.437,
    "VEV_5300": 0.273,
    "VEV_5400": 0.129,
    "VEV_5500": 0.055,
    "VEV_6000": 0.0,
    "VEV_6500": 0.0,
}


def _half_spread(bid1: str, ask1: str) -> float | None:
    try:
        b, a = int(bid1), int(ask1)
    except (TypeError, ValueError):
        return None
    if a <= b:
        return None
    return 0.5 * (a - b)


def main() -> None:
    by_sym: dict[str, list[float]] = defaultdict(list)

    for d in DAYS:
        path = ROOT / f"prices_round_3_day_{d}.csv"
        rows_by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                ts = int(row["timestamp"])
                p = row["product"]
                if p not in SYMS and p != UNDER:
                    continue
                rows_by_ts[ts][p] = row
        for ts in sorted(rows_by_ts):
            m = rows_by_ts[ts]
            if UNDER not in m:
                continue
            row_u = m[UNDER]
            for s in SYMS:
                if s not in m:
                    continue
                ro = m[s]
                h = _half_spread(ro["bid_price_1"], ro["ask_price_1"])
                if h is None:
                    continue
                by_sym[s].append(h)
            u_mids.append((ts, su))

    out: dict = {"method": "L1 book half-spread = 0.5*(ask-bid) per tick; VELVETFRUIT_EXTRACT for context only", "n_ticks_with_row": {s: len(v) for s, v in by_sym.items()}}
    out["by_symbol"] = {}
    for s in SYMS:
        arr = by_sym.get(s) or [0.0]
        mean_h = sum(arr) / len(arr)
        sorted_h = sorted(arr)
        p50 = sorted_h[len(sorted_h) // 2]
        p90 = sorted_h[int(0.9 * (len(sorted_h) - 1))]
        b = BETA_TAPE[s]
        out["by_symbol"][s] = {
            "mean_half_spread": mean_h,
            "p50_half_spread": p50,
            "p90_half_spread": p90,
            "beta_tape_ols": b,
            "opportunity_proxy_beta_over_mean_h": (b / mean_h) if mean_h > 1e-9 else 0.0,
        }

    out["interpretation"] = "Higher beta at low strikes coexists with much wider mean half-spread; weighting by beta/h favors ATM-like VEV_5000-5200 for same model edge."
    Path(__file__).resolve().parent.joinpath("analysis_spread_elasticity_r3.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote analysis_spread_elasticity_r3.json")


if __name__ == "__main__":
    main()
