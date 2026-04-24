"""
Validate osmium_inner_bot.py against fair-probe CSV + osmium_true_fv.csv.

Use --all-sessions to validate every folder in osmium_sessions.all_session_dirs()
and print a pooled summary (both fair exports).
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from osmium_fair_common import DEFAULT_DATA_DIR, load_ticks
from osmium_inner_bot import inner_prices
from osmium_sessions import all_session_dirs


def chi2_uniform(counts: Counter[int], lo: int, hi: int, n: int) -> float:
    k = hi - lo + 1
    exp = n / k
    return sum((counts.get(v, 0) - exp) ** 2 / exp for v in range(lo, hi + 1))


def validate_one(data_dir: Path) -> None:
    ticks, prices_path = load_ticks(data_dir)
    n = len(ticks)

    print("=" * 70)
    print("  VALIDATION: osmium_inner_bot (round(FV) - 8 / + 8)")
    print(f"  data_dir={data_dir.resolve()}")
    print(f"  prices={prices_path.name}  timesteps={n}")
    print("=" * 70)

    bid_match = ask_match = both_match = spread_match = 0
    bid_errs: Counter[int] = Counter()
    ask_errs: Counter[int] = Counter()

    for t in ticks:
        sb, sa = inner_prices(t.fv)
        ib, ia = t.inner_bid, t.inner_ask
        if ib is not None:
            bm = sb == ib
            bid_match += int(bm)
            if not bm:
                bid_errs[sb - ib] += 1
        if ia is not None:
            am = sa == ia
            ask_match += int(am)
            if not am:
                ask_errs[sa - ia] += 1
        if ib is not None and ia is not None:
            both_match += int(sb == ib and sa == ia)
            spread_match += int((sa - sb) == (ia - ib))

    n_ib = sum(1 for t in ticks if t.inner_bid is not None)
    n_ia = sum(1 for t in ticks if t.inner_ask is not None)
    n_both = sum(1 for t in ticks if t.inner_bid is not None and t.inner_ask is not None)

    print(f"\n  Ticks with inner bid (−8): {n_ib}/{n}")
    print(f"  Ticks with inner ask (+8): {n_ia}/{n}")
    print(f"  Ticks with both inner legs: {n_both}/{n}")
    print(f"\n  Bid match:    {bid_match}/{n_ib} ({100 * bid_match / max(n_ib, 1):.1f}%)")
    print(f"  Ask match:    {ask_match}/{n_ia} ({100 * ask_match / max(n_ia, 1):.1f}%)")
    print(f"  Both match:   {both_match}/{n_both} ({100 * both_match / max(n_both, 1):.1f}%)")
    print(f"  Spread match: {spread_match}/{n_both} ({100 * spread_match / max(n_both, 1):.1f}%)")
    if bid_errs:
        print(f"\n  Bid errors (pred - actual): {dict(bid_errs)}")
    if ask_errs:
        print(f"  Ask errors (pred - actual): {dict(ask_errs)}")

    bv: Counter[int] = Counter()
    av: Counter[int] = Counter()
    same_vol = 0
    n_vol_pair = 0
    for t in ticks:
        sb, sa = inner_prices(t.fv)
        if t.inner_bid is not None and sb == t.inner_bid and t.inner_bid_vol is not None:
            bv[t.inner_bid_vol] += 1
        if t.inner_ask is not None and sa == t.inner_ask and t.inner_ask_vol is not None:
            av[t.inner_ask_vol] += 1
        if (
            t.inner_bid is not None
            and t.inner_ask is not None
            and sb == t.inner_bid
            and sa == t.inner_ask
            and t.inner_bid_vol is not None
            and t.inner_ask_vol is not None
        ):
            n_vol_pair += 1
            if t.inner_bid_vol == t.inner_ask_vol:
                same_vol += 1

    nbv = sum(bv.values())
    print(f"\n  Bid vol @ price match: n={nbv}  chi2 U[10..15]={chi2_uniform(bv, 10, 15, nbv):.2f}" if nbv else "")
    nav = sum(av.values())
    if nav:
        print(f"  Ask vol @ price match: n={nav}  chi2 U[10..15]={chi2_uniform(av, 10, 15, nav):.2f}")
    print(f"  Bid vol == ask vol (both match): {same_vol}/{n_vol_pair}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate inner osmium MM vs fair CSV")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--all-sessions", action="store_true")
    args = ap.parse_args()

    if args.all_sessions:
        dirs = all_session_dirs()
    elif args.data_dir is not None:
        dirs = (args.data_dir,)
    else:
        dirs = (DEFAULT_DATA_DIR,)

    pool_ib = pool_ib_n = pool_ia = pool_ia_n = pool_both = pool_both_n = 0
    for d in dirs:
        ticks, _ = load_ticks(d)
        bm = am = bot = sp = 0
        n_ib = sum(1 for t in ticks if t.inner_bid is not None)
        n_ia = sum(1 for t in ticks if t.inner_ask is not None)
        n_both = sum(1 for t in ticks if t.inner_bid is not None and t.inner_ask is not None)
        for t in ticks:
            sb, sa = inner_prices(t.fv)
            ib, ia = t.inner_bid, t.inner_ask
            if ib is not None:
                bm += int(sb == ib)
            if ia is not None:
                am += int(sa == ia)
            if ib is not None and ia is not None:
                bot += int(sb == ib and sa == ia)
                sp += int((sa - sb) == (ia - ib))
        pool_ib += bm
        pool_ib_n += n_ib
        pool_ia += am
        pool_ia_n += n_ia
        pool_both += bot
        pool_both_n += n_both
        validate_one(d)

    if len(dirs) > 1:
        print("=" * 70)
        print("  POOLED (all sessions)")
        print("=" * 70)
        print(
            f"  Bid match:  {pool_ib}/{pool_ib_n} ({100 * pool_ib / max(pool_ib_n, 1):.1f}%)\n"
            f"  Ask match:  {pool_ia}/{pool_ia_n} ({100 * pool_ia / max(pool_ia_n, 1):.1f}%)\n"
            f"  Both match: {pool_both}/{pool_both_n} ({100 * pool_both / max(pool_both_n, 1):.1f}%)"
        )


if __name__ == "__main__":
    main()
