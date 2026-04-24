#!/usr/bin/env python3
"""
Wall bot: same spirit as hiddenalphastuff/bot1_exact_rule.py.

1) Price anchor: for each rung, implied R = bid+|offset| or ask-|offset|
   should match round(FV) (tomato-style single integer anchor).

2) "−10 vs −11": both are **fixed ticks from the same R** — not two different
   rounding rules. The open question is **which rungs post** (presence), not
   two competing price formulas.

3) Presence vs FV fractional part (pooled across sessions when --all-sessions).
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from osmium_fair_common import DEFAULT_DATA_DIR, load_ticks
from osmium_sessions import all_session_dirs


def frac_bin05(fv: float) -> float:
    f = fv - math.floor(fv)
    b = round(f / 0.05) * 0.05
    if b >= 1.0:
        b = 0.0
    return b


def analyze_ticks(ticks: list, label: str) -> None:
    print("=" * 70)
    print(f"  OSMIUM WALL — EXACT ANCHOR + PRESENCE  ({label})")
    print("=" * 70)

    # Same R when both bid rungs present
    both_b = [
        t
        for t in ticks
        if t.bid_m10 is not None and t.bid_m11 is not None
    ]
    same_r_b = sum(
        1
        for t in both_b
        if (t.bid_m10 + 10 == t.bid_m11 + 11 == round(t.fv))
    )
    print(f"\n  Bid: ticks with BOTH −10 and −11: {len(both_b)}")
    if both_b:
        print(
            f"    Of those, bid_m10+10 == bid_m11+11 == round(FV): "
            f"{same_r_b}/{len(both_b)} ({100*same_r_b/len(both_b):.1f}%)"
        )

    both_a = [
        t
        for t in ticks
        if t.ask_p10 is not None and t.ask_p11 is not None
    ]
    same_r_a = sum(
        1
        for t in both_a
        if (t.ask_p10 - 10 == t.ask_p11 - 11 == round(t.fv))
    )
    print(f"  Ask: ticks with BOTH +10 and +11: {len(both_a)}")
    if both_a:
        print(
            f"    ask_p10-10 == ask_p11-11 == round(FV): "
            f"{same_r_a}/{len(both_a)} ({100*same_r_a/len(both_a):.1f}%)"
        )

    # Per-rung: R vs round(FV) misses
    def misses(getter, k: int, side: str) -> Counter[int]:
        c: Counter[int] = Counter()
        for t in ticks:
            p = getter(t)
            if p is None:
                continue
            if side == "bid":
                r_implied = p + abs(k)
            else:
                r_implied = p - abs(k)
            d = r_implied - round(t.fv)
            if d != 0:
                c[d] += 1
        return c

    for name, getter, k, side in [
        ("bid_m10", lambda t: t.bid_m10, -10, "bid"),
        ("bid_m11", lambda t: t.bid_m11, -11, "bid"),
        ("ask_p10", lambda t: t.ask_p10, 10, "ask"),
        ("ask_p11", lambda t: t.ask_p11, 11, "ask"),
    ]:
        m = misses(getter, k, side)
        tot = sum(m.values())
        nv = sum(1 for t in ticks if getter(t) is not None)
        print(f"\n  {name}: implied R misses vs round(FV): {tot} / {nv} present")

    # R_b10 = bid_m10+10 distribution vs fv_frac (when m10 present)
    print(f"\n  R_b10 = bid_m10 + 10  vs floor(FV) offset (0.05 frac bins, when −10 present):")
    frac_r: dict[float, list[float]] = defaultdict(list)
    for t in ticks:
        if t.bid_m10 is None:
            continue
        fv = t.fv
        r = t.bid_m10 + 10
        fb = frac_bin05(fv)
        frac_r[fb].append(r - math.floor(fv))
    for f in sorted(frac_r)[:12]:
        ctr = Counter(frac_r[f])
        s = ", ".join(f"{v}:{c}" for v, c in sorted(ctr.items()))
        print(f"    frac~{f:.2f}  {s}")

    # Presence vs frac (all ticks)
    n = len(ticks)
    pres = defaultdict(lambda: {"n": 0, "b10": 0, "b11": 0, "a10": 0, "a11": 0})
    for t in ticks:
        fb = frac_bin05(t.fv)
        pres[fb]["n"] += 1
        if t.bid_m10 is not None:
            pres[fb]["b10"] += 1
        if t.bid_m11 is not None:
            pres[fb]["b11"] += 1
        if t.ask_p10 is not None:
            pres[fb]["a10"] += 1
        if t.ask_p11 is not None:
            pres[fb]["a11"] += 1

    print(f"\n  PRESENCE vs FV fractional bin (first 12 bins; n = ticks in bin):")
    print(f"    {'frac':>6} {'n':>5} {'b10%':>7} {'b11%':>7} {'a10%':>7} {'a11%':>7}")
    for fb in sorted(pres)[:12]:
        d = pres[fb]
        nn = d["n"]
        if nn == 0:
            continue
        print(
            f"    {fb:>6.2f} {nn:>5} "
            f"{100*d['b10']/nn:>6.1f}% {100*d['b11']/nn:>6.1f}% "
            f"{100*d['a10']/nn:>6.1f}% {100*d['a11']/nn:>6.1f}%"
        )

    print(
        f"\n  INTERPRETATION: If −10 vs −11 were chosen by different *price* rules, "
        f"R_b10 and R_b11 would disagree when both post. When both exist, they match "
        f"the same round(FV). Choosing **which depth to display** is separate "
        f"(partial ladder / scheduling); marginal presence vs frac above is for "
        f"exploration only (thin n per bin on 1k ticks)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument(
        "--all-sessions",
        action="store_true",
        help="Run 278076 + r2_osmium_fair_248329",
    )
    args = ap.parse_args()
    if args.all_sessions:
        dirs = all_session_dirs()
    elif args.data_dir is not None:
        dirs = (args.data_dir,)
    else:
        dirs = (DEFAULT_DATA_DIR,)

    pooled: list = []
    for d in dirs:
        ticks, _ = load_ticks(d)
        analyze_ticks(ticks, str(d.resolve()))
        pooled.extend(ticks)
    if len(dirs) > 1:
        analyze_ticks(pooled, "POOLED (all sessions)")


if __name__ == "__main__":
    main()
