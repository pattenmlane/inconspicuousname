#!/usr/bin/env python3
"""
Round-3 tape: when |log-return| of VELVETFRUIT_EXTRACT mid exceeds quantiles,
how much does each VEV's L1 spread (ask1-bid1) widen vs baseline?
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path("Prosperity4Data/ROUND_3")
U = "VELVETFRUIT_EXTRACT"
VEVS = [f"VEV_{k}" for k in (5000, 5100, 5200, 5300, 5400, 5500)]
OUT = ROOT / "analysis_outputs" / "tape_extract_shock_spread_widen.json"


def load_by_ts(path: Path) -> dict[int, dict[str, dict]]:
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            ts = int(r["timestamp"])
            prod = r["product"]
            by_ts[ts][prod] = r
    return dict(by_ts)


def spread1(r: dict) -> int | None:
    try:
        return int(float(r["ask_price_1"])) - int(float(r["bid_price_1"]))
    except (KeyError, ValueError, TypeError):
        return None


def main() -> None:
    out: dict = {"method": "Consecutive timestamps; r = |log(S_t/S_{t-1})| for extract mid.", "by_day": {}}
    for d in (0, 1, 2):
        by_ts = load_by_ts(DATA / f"prices_round_3_day_{d}.csv")
        tss = sorted(by_ts.keys())
        rets: list[float] = []
        prev_s: float | None = None
        for ts in tss:
            row = by_ts[ts].get(U)
            if row is None:
                continue
            s = float(row["mid_price"])
            if prev_s is not None and prev_s > 0 and s > 0:
                rets.append(abs(math.log(s / prev_s)))
            prev_s = s
        if not rets:
            continue
        rets_sorted = sorted(rets)
        q90 = rets_sorted[int(0.90 * (len(rets_sorted) - 1))]
        q95 = rets_sorted[int(0.95 * (len(rets_sorted) - 1))]

        per_sym: dict = {}
        for sym in VEVS:
            base_sp: list[int] = []
            shock_sp: list[int] = []
            prev_s = None
            prev_spread: int | None = None
            for ts in tss:
                ur = by_ts[ts].get(U)
                vr = by_ts[ts].get(sym)
                if ur is None or vr is None:
                    continue
                s = float(ur["mid_price"])
                sp = spread1(vr)
                if sp is None:
                    continue
                if prev_s is not None and prev_s > 0 and s > 0 and prev_spread is not None:
                    r = abs(math.log(s / prev_s))
                    if r >= q90:
                        shock_sp.append(sp)
                    else:
                        base_sp.append(sp)
                prev_s = s
                prev_spread = sp

            def med(xs: list[int]) -> float | None:
                if not xs:
                    return None
                return float(statistics.median(xs))

            bm, sm = med(base_sp), med(shock_sp)
            per_sym[sym] = {
                "n_base": len(base_sp),
                "n_shock90": len(shock_sp),
                "spread_median_base": bm,
                "spread_median_shock90": sm,
                "median_widen_ticks": (sm - bm) if bm is not None and sm is not None else None,
            }
        out["by_day"][str(d)] = {
            "abs_logret_p90": q90,
            "abs_logret_p95": q95,
            "per_symbol": per_sym,
        }
    OUT.write_text(json.dumps(out, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
