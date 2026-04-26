#!/usr/bin/env python3
"""
v26 trigger windows on merged timeline: interval [T, T+W] where signal is active
(active iff exists trigger tr with ts-W < tr <= ts, i.e. ts in [T, T+W] for trigger at T).

Computes pairwise gaps, max simultaneous overlaps (sweep), and merged coverage length.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
SIG = OUT / "r4_v26_signals.json"


def main() -> None:
    obj = json.loads(SIG.read_text())
    trigs = sorted(int(x) for x in obj["mark67_extract_buy_aggr_filtered_merged_ts"])
    W = int(obj["window_ts"])
    cum = {int(k): int(v) for k, v in obj["day_cum_offset"].items()}

    def to_day(t: int) -> int:
        for d in sorted(cum.keys(), reverse=True):
            if t >= cum[d]:
                return d
        return 1

    # Active at ts iff exists trigger T with (ts-W) <= T <= ts (matches bisect in trader_v26).
    # For fixed T: ts in [T, T+W]. Half-open sweep [T, T+W+1).
    intervals = [(t, t + W + 1) for t in trigs]
    gaps = []
    for a, b in zip(trigs, trigs[1:]):
        gaps.append(b - a)
    events: list[tuple[int, int]] = []
    for lo, hi in intervals:
        events.append((lo, 1))
        events.append((hi, -1))
    events.sort()
    cur = 0
    mx = 0
    for _, d in events:
        cur += d
        mx = max(mx, cur)
    # merged length (union)
    merged = 0
    if intervals:
        iv = sorted(intervals)
        ca, cb = iv[0]
        for lo, hi in iv[1:]:
            if lo <= cb:
                cb = max(cb, hi)
            else:
                merged += cb - ca
                ca, cb = lo, hi
        merged += cb - ca

    raw_len = len(trigs) * (W + 1)
    lines = [
        f"n_triggers={len(trigs)} W={W} (active ts interval length per trigger = W+1 in discrete inclusive model)",
        f"max_simultaneous_active_windows={mx}",
        f"sum_halfopen_lengths={raw_len} merged_union_length={merged} overlap_reduction={raw_len - merged}",
    ]
    if gaps:
        lines.append(
            f"gap_start_to_start: min={min(gaps)} median={int(statistics.median(gaps))} max={max(gaps)}"
        )
    by_day: dict[int, list[int]] = {}
    for t in trigs:
        by_day.setdefault(to_day(t), []).append(t)
    lines.append("triggers_per_merged_day:")
    for d in sorted(by_day):
        lines.append(f"  day_{d}: n={len(by_day[d])}")
    outp = OUT / "r4_p8_v26_window_overlap.txt"
    outp.write_text("\n".join(lines) + "\n")
    print("wrote", outp)


if __name__ == "__main__":
    main()
