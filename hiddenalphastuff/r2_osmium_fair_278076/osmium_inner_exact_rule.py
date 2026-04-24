"""
Examine inner-bot rounding: R_bid = inner_bid + 8, R_ask = inner_ask - 8 vs FV.

Analogous to hiddenalphastuff/bot1_exact_rule.py for tomato wall bot.
Only uses ticks where the inner bid (resp. ask) level is present.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict

from osmium_fair_common import DEFAULT_DATA_DIR, load_ticks
from osmium_sessions import all_session_dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--all-sessions", action="store_true")
    args = ap.parse_args()
    if args.all_sessions:
        dirs = all_session_dirs()
    elif args.data_dir is not None:
        dirs = (args.data_dir,)
    else:
        dirs = (DEFAULT_DATA_DIR,)

    for data_dir in dirs:
        ticks, _ = load_ticks(data_dir)
        _run_inner_exact(ticks, str(data_dir.resolve()))


def _run_inner_exact(ticks: list, data_dir_label: str) -> None:

    records_b = []
    records_a = []
    for t in ticks:
        if t.inner_bid is not None:
            records_b.append({"fv": t.fv, "bid": t.inner_bid})
        if t.inner_ask is not None:
            records_a.append({"fv": t.fv, "ask": t.inner_ask})

    print("=" * 70)
    print("  OSMIUM INNER — EXACT ROUNDING ANALYSIS")
    print(f"  data_dir={data_dir_label}")
    print(f"  samples bid: {len(records_b)}  ask: {len(records_a)}")
    print("=" * 70)

    print("\n  R_bid = inner_bid + 8  (should equal round(FV) when model holds)")
    print(f"  {'fv_frac':>8} {'R_bid - floor(FV)':>20} (counts)")
    frac_data_b: dict[float, list[float]] = defaultdict(list)
    for rec in records_b:
        fv = rec["fv"]
        r_bid = rec["bid"] + 8
        frac = fv - math.floor(fv)
        frac_bin = round(frac * 20) / 20
        if frac_bin >= 1.0:
            frac_bin = 0.0
        frac_data_b[frac_bin].append(r_bid - math.floor(fv))

    for f in sorted(frac_data_b):
        ctr = Counter(frac_data_b[f])
        s = ", ".join(f"{v}:{c}" for v, c in sorted(ctr.items()))
        print(f"  {f:>8.2f}  {s}")

    print("\n  R_ask = inner_ask - 8")
    frac_data_a: dict[float, list[float]] = defaultdict(list)
    for rec in records_a:
        fv = rec["fv"]
        r_ask = rec["ask"] - 8
        frac = fv - math.floor(fv)
        frac_bin = round(frac * 20) / 20
        if frac_bin >= 1.0:
            frac_bin = 0.0
        frac_data_a[frac_bin].append(r_ask - math.floor(fv))

    for f in sorted(frac_data_a):
        ctr = Counter(frac_data_a[f])
        s = ", ".join(f"{v}:{c}" for v, c in sorted(ctr.items()))
        print(f"  {f:>8.2f}  {s}")

    print(f"\n{'=' * 70}")
    print("  MISS ANALYSIS: actual - round(FV) prediction")
    print("=" * 70)

    miss_b = Counter()
    for rec in records_b:
        d = rec["bid"] - (round(rec["fv"]) - 8)
        if d != 0:
            miss_b[d] += 1
    print(f"  BID (inner): misses {sum(miss_b.values())}  by delta: {dict(miss_b)}")

    miss_a = Counter()
    for rec in records_a:
        d = rec["ask"] - (round(rec["fv"]) + 8)
        if d != 0:
            miss_a[d] += 1
    print(f"  ASK (inner): misses {sum(miss_a.values())}  by delta: {dict(miss_a)}")

    print(f"\n{'=' * 70}")
    print("  When FV fractional part near 0.5, list first few bid misses")
    print("=" * 70)
    shown = 0
    for rec in records_b:
        fv = rec["fv"]
        pred = round(fv) - 8
        if rec["bid"] == pred:
            continue
        frac = fv - math.floor(fv)
        if 0.45 <= frac <= 0.55 or fv - round(fv) < 0.05:
            print(f"    fv={fv:.6f} frac={frac:.4f} bid={rec['bid']} pred={pred}")
            shown += 1
            if shown >= 12:
                break
    if shown == 0:
        print("    (none in boundary window on this slice)")


if __name__ == "__main__":
    main()

