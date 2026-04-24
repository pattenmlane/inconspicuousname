#!/usr/bin/env python3
"""
Near-FV layer: exact-structure checks (tomato bot1_exact_rule spirit).

Uses |price - true_FV| <= 4 events. Reports:
  - delta = price - round(FV) distribution and rare deltas
  - implied anchor check (tautology check)
  - fv fractional bin x delta coarse heatmap counts
  - crossing vs delta
  - small-sample caveat

--all-sessions: run each session dir then pooled merged events for joint stats.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from osmium_near_fv_events import (
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_ABS,
    iter_near_fv_events,
    load_rows_and_fv,
)
from osmium_sessions import all_session_dirs


def frac_bin05(fv: float) -> float:
    f = fv - math.floor(fv)
    b = round(f / 0.05) * 0.05
    if b >= 1.0:
        b = 0.0
    return b


def run_session(data_dir: Path, max_abs: float, label: str) -> list:
    rows, fv_map, _ = load_rows_and_fv(data_dir)
    ev = list(iter_near_fv_events(rows, fv_map, max_abs=max_abs, exclude_mm_offsets=False))

    print("=" * 70)
    print(f"  OSMIUM NEAR-FV — EXACT STRUCTURE  ({label})")
    print(f"  events={len(ev)}  |price-FV|<={max_abs}")
    print("=" * 70)

    # delta = price - round(fv) — by construction price = round(fv) + delta for integer delta
    d_ct = Counter(e.delta for e in ev)
    print("\n  delta = price - round(FV):")
    for d in sorted(d_ct):
        print(f"    {d:+d}: {d_ct[d]}")

    rare = [e for e in ev if e.delta not in {-3, -2, -1, 0, 1, 2, 3}]
    if rare:
        print(f"\n  Rare |delta|>3 or unusual (n={len(rare)}):")
        for e in rare[:15]:
            print(
                f"    ts={e.timestamp} side={e.side} price={e.price} fv={e.fv:.4f} "
                f"delta={e.delta} off_cont={e.off_cont:.3f} crossing={e.crossing}"
            )

    # price == round(fv) + delta always for integer price
    bad = [e for e in ev if e.price != round(e.fv) + e.delta]
    print(f"\n  Tautology check price == round(FV)+delta: fails {len(bad)} / {len(ev)}")

    print("\n  fv_frac (0.05) x delta (coarse) counts [row=frac_bin, col=delta]:")
    grid: dict[tuple[float, int], int] = defaultdict(int)
    for e in ev:
        grid[(frac_bin05(e.fv), e.delta)] += 1
    deltas = sorted({e.delta for e in ev})
    fracs = sorted({frac_bin05(e.fv) for e in ev})[:14]
    hdr = "frac\\d " + "".join(f"{d:>5}" for d in deltas)
    print(f"    {hdr}")
    for fb in fracs:
        row = f"    {fb:>5.2f}"
        for d in deltas:
            row += f"{grid.get((fb, d), 0):>5}"
        print(row)

    print("\n  crossing x delta:")
    j = Counter((e.crossing, e.delta) for e in ev)
    for k in sorted(j):
        print(f"    {k}: {j[k]}")

    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--max-abs", type=float, default=DEFAULT_MAX_ABS)
    ap.add_argument("--all-sessions", action="store_true")
    args = ap.parse_args()
    if args.all_sessions:
        dirs = all_session_dirs()
    elif args.data_dir is not None:
        dirs = (args.data_dir,)
    else:
        dirs = (DEFAULT_DATA_DIR,)

    pooled: list = []
    for d in dirs:
        pooled.extend(run_session(d, args.max_abs, str(d.resolve())))

    if len(dirs) > 1:
        print("=" * 70)
        print("  POOLED — merged near-FV events (both sessions)")
        print("=" * 70)
        d_ct = Counter(e.delta for e in pooled)
        print(f"  n={len(pooled)}  delta counts: {dict(sorted(d_ct.items()))}")
        ring = {-3, -2, 1, 2}
        sub = [e for e in pooled if e.delta in ring]
        if sub:
            ct = Counter(e.delta for e in sub)
            exp = len(sub) / 4
            chi = sum((ct.get(d, 0) - exp) ** 2 / exp for d in sorted(ring))
            print(f"  ring {{-3,-2,+1,+2}} chi2 vs uniform: {chi:.3f} df=3 n={len(sub)}")
        cr = sum(1 for e in pooled if e.crossing)
        print(f"  crossing: {cr}/{len(pooled)} ({100*cr/max(len(pooled),1):.1f}%)")

    print(
        "\n  Note: near-FV is sparse (~80 events / 1k ticks); "
        "fraction-delta grid has small counts per cell — exploratory only."
    )


if __name__ == "__main__":
    main()
