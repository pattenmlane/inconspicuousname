#!/usr/bin/env python3
"""Per-strike top-of-book depth and spread from Round 3 price tapes (VEV vouchers).

Uses bid_price_1, bid_volume_1, ask_price_1, ask_volume_1 (sells are negative in tape;
we take abs for depth). Also aggregates bid_price_2/3 and ask_2/3 when non-empty
for a simple multi-level depth sum.

Output: analysis_outputs/vev_book_depth_by_strike.csv
Methodology: logged in analysis.json; informs dynamic MM/TAKE sizing vs fixed v11.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
VEVS = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]


def f(v: str) -> float | None:
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def iabs(v: str) -> int:
    x = f(v)
    if x is None:
        return 0
    return int(abs(x))


def row_depths(row: dict[str, str]) -> tuple[int, int, int, float | None]:
    """Return (top_bid_vol, top_ask_vol, sum_levels_1_to_3_bid+ask, spread or None)."""
    bb = f(row.get("bid_price_1", ""))
    ba = f(row.get("ask_price_1", ""))
    if bb is None or ba is None:
        return 0, 0, 0, None
    spr = float(ba - bb)
    b1, a1 = iabs(row.get("bid_volume_1", "")), iabs(row.get("ask_volume_1", ""))
    b2, a2 = iabs(row.get("bid_volume_2", "")), iabs(row.get("ask_volume_2", ""))
    b3, a3 = iabs(row.get("bid_volume_3", "")), iabs(row.get("ask_volume_3", ""))
    sum_ba = b1 + a1 + b2 + a2 + b3 + a3
    return b1, a1, sum_ba, spr


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # per (day, product): list of (min_top, sum_top, spread, imbalance)
    buckets: dict[tuple[int, str], list[tuple[int, int, float, float]]] = defaultdict(list)

    for day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{day}.csv"
        with path.open(newline="") as fp:
            for r in csv.DictReader(fp, delimiter=";"):
                p = r.get("product", "")
                if p not in VEVS:
                    continue
                b1, a1, sum_ba, spr = row_depths(r)
                if spr is None or spr < 0 or math.isnan(spr):
                    continue
                m = min(b1, a1) if b1 and a1 else 0
                imb = 0.0
                if b1 + a1 > 0:
                    imb = (b1 - a1) / float(b1 + a1)
                buckets[(day, p)].append((m, sum_ba, float(spr), imb))

    rows_out: list[dict[str, str | float | int]] = []
    for (day, sym), vals in sorted(buckets.items()):
        mins = [v[0] for v in vals]
        sums = [v[1] for v in vals]
        sprs = [v[2] for v in vals]
        imbs = [v[3] for v in vals]
        n = len(vals)
        rows_out.append(
            {
                "day": day,
                "product": sym,
                "n_rows": n,
                "min_top_of_book": float(np.mean(mins)) if n else 0.0,
                "median_min_top": float(np.median(mins)) if n else 0.0,
                "mean_spread": float(np.mean(sprs)) if n else 0.0,
                "median_spread": float(np.median(sprs)) if n else 0.0,
                "mean_sum_depth_3lv": float(np.mean(sums)) if n else 0.0,
                "mean_abs_imbalance": float(np.mean(np.abs(imbs))) if n else 0.0,
            }
        )

    # Pooled over days per strike
    by_sym: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for (day, sym), vals in buckets.items():
        for m, sba, spr, _ in vals:
            by_sym[sym].append((m, sba, spr))

    for sym in VEVS:
        vals = by_sym.get(sym, [])
        if not vals:
            continue
        mins = [v[0] for v in vals]
        sums = [v[1] for v in vals]
        sprs = [v[2] for v in vals]
        n = len(vals)
        rows_out.append(
            {
                "day": -1,
                "product": sym,
                "n_rows": n,
                "min_top_of_book": float(np.mean(mins)),
                "median_min_top": float(np.median(mins)),
                "mean_spread": float(np.mean(sprs)),
                "median_spread": float(np.median(sprs)),
                "mean_sum_depth_3lv": float(np.mean(sums)),
                "mean_abs_imbalance": 0.0,
            }
        )

    out_path = OUT / "vev_book_depth_by_strike.csv"
    with out_path.open("w", newline="") as fp:
        w = csv.DictWriter(
            fp,
            fieldnames=[
                "day",
                "product",
                "n_rows",
                "min_top_of_book",
                "median_min_top",
                "mean_spread",
                "median_spread",
                "mean_sum_depth_3lv",
                "mean_abs_imbalance",
            ],
        )
        w.writeheader()
        w.writerows(rows_out)

    # Console summary for v11 baseline sizing (18/22)
    pool = []
    for sym in VEVS:
        pool.extend(by_sym.get(sym, []))
    if pool:
        mns = [p[0] for p in pool]
        print(
            f"Pooled all VEV: mean min(top bid, top ask)={float(np.mean(mns)):.2f}, "
            f"median={float(np.median(mns)):.2f} (n={len(pool)})"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
