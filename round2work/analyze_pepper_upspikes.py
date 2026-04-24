#!/usr/bin/env python3
"""
Tick-to-tick **up** moves in pepper **touch mid** (CSV ``mid_price``), Prosperity4Data.

Spike up: ``mid[t] - mid[t-1] > 0`` on consecutive pepper rows within each CSV.

**RAW** includes bad rows (e.g. ``mid_price == 0`` when the book is empty) → huge fake “spikes”.

**CLEAN** keeps only pairs where both mids are in ``[MID_LO, MID_HI]`` (normal pepper range).

Run from repo root::

  python3 round2work/analyze_pepper_upspikes.py
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "Prosperity4Data"
PEPPER = "INTARIAN_PEPPER_ROOT"

MID_LO = 5000.0
MID_HI = 25_000.0

BINS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, float("inf"))
BIN_LABELS = [
    "(0, 0.5]",
    "(0.5, 1]",
    "(1, 2]",
    "(2, 3]",
    "(3, 5]",
    "(5, 10]",
    "(10, 20]",
    "(20, 50]",
    "> 50",
]


def bin_upward(delta: float) -> int:
    for i, hi in enumerate(BINS):
        if delta <= hi:
            return i
    return len(BINS) - 1


def iter_price_files(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for p in sorted(sub.glob("prices_round_*_day_*.csv")):
            if "_enriched" in p.name:
                continue
            out.append(p)
    return sorted(set(out))


def mids_from_csv(path: Path) -> list[float]:
    mids: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";")
        next(r, None)
        for row in r:
            if len(row) < 17:
                continue
            if row[2] != PEPPER:
                continue
            mids.append(float(row[15]))
    return mids


def pctile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = int(math.ceil((len(xs) - 1) * p / 100.0))
    return xs[min(i, len(xs) - 1)]


def print_block(
    title: str,
    deltas: list[float],
    bin_counts: Counter,
    n_pairs: int,
    n_up: int,
    per_file_max: list[tuple[str, float]],
) -> None:
    print(title)
    print(f"  consecutive pepper mids (pairs): {n_pairs:,}")
    print(f"  strictly **up** steps: {n_up:,} ({100 * n_up / max(1, n_pairs):.2f}% of pairs)")
    if not deltas:
        print("  (no upward steps)")
        print()
        return
    print("  upward jump size (price points):")
    print(f"    min: {min(deltas):.4g}")
    print(f"    p50: {pctile(deltas, 50):.4g}")
    print(f"    p90: {pctile(deltas, 90):.4g}")
    print(f"    p95: {pctile(deltas, 95):.4g}")
    print(f"    p99: {pctile(deltas, 99):.4g}")
    print(f"    max: {max(deltas):.4g}")
    print("  histogram (count of **up** steps in bin):")
    for i, lab in enumerate(BIN_LABELS):
        c = bin_counts.get(i, 0)
        pc = 100 * c / max(1, n_up)
        bar = "#" * int(pc / 0.5)
        print(f"    {lab:>12}  {c:>8,}  ({pc:5.2f}%)  {bar}")
    top = sorted(per_file_max, key=lambda t: -t[1])[:8]
    print("  max single-tick **up** per file (top 8):")
    for rel, mx in top:
        print(f"    {mx:>8.2f}  {rel}")
    print()


def main() -> None:
    files = iter_price_files(DATA)
    if not files:
        raise SystemExit(f"No prices CSVs under {DATA}")

    raw_deltas: list[float] = []
    raw_bins = Counter()
    raw_pairs = 0
    raw_up = 0
    raw_max_per_file: list[tuple[str, float]] = []

    clean_deltas: list[float] = []
    clean_bins = Counter()
    clean_pairs = 0
    clean_up = 0
    clean_max_per_file: list[tuple[str, float]] = []

    for path in files:
        rel = str(path.relative_to(DATA))
        mids = mids_from_csv(path)
        if len(mids) < 2:
            continue
        loc_raw = 0.0
        loc_clean = 0.0
        for a, b in zip(mids, mids[1:]):
            raw_pairs += 1
            d = b - a
            if d > 0:
                raw_up += 1
                raw_deltas.append(d)
                raw_bins[bin_upward(d)] += 1
                loc_raw = max(loc_raw, d)

            if MID_LO <= a <= MID_HI and MID_LO <= b <= MID_HI:
                clean_pairs += 1
                if d > 0:
                    clean_up += 1
                    clean_deltas.append(d)
                    clean_bins[bin_upward(d)] += 1
                    loc_clean = max(loc_clean, d)

        raw_max_per_file.append((rel, loc_raw))
        clean_max_per_file.append((rel, loc_clean))

    print(f"data root: {DATA}")
    print(f"product: {PEPPER}")
    print(f"files: {len(files)}")
    print(f"valid mid band (CLEAN): [{MID_LO:g}, {MID_HI:g}]")
    print()
    print_block(
        "--- RAW (mid can be 0 → huge fake spikes) ---",
        raw_deltas,
        raw_bins,
        raw_pairs,
        raw_up,
        raw_max_per_file,
    )
    print_block(
        f"--- CLEAN (both mids in [{MID_LO:g}, {MID_HI:g}]) ---",
        clean_deltas,
        clean_bins,
        clean_pairs,
        clean_up,
        clean_max_per_file,
    )


if __name__ == "__main__":
    main()
