"""
Validate osmium_wall_bot.py per rung. Use --all-sessions for both fair exports
+ pooled price-match and pooled volume χ² (merged counts).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from osmium_fair_common import DEFAULT_DATA_DIR, load_ticks
from osmium_sessions import all_session_dirs
from osmium_wall_bot import wall_prices


def chi2_uniform(counts: Counter[int], lo: int, hi: int, n: int) -> float:
    k = hi - lo + 1
    exp = n / k
    return sum((counts.get(v, 0) - exp) ** 2 / exp for v in range(lo, hi + 1))


def leg_report(
    ticks,
    actual_price,
    actual_vol,
    pred_price_fn,
) -> tuple[int, int, Counter[int]]:
    match = 0
    vol_ct: Counter[int] = Counter()
    n = 0
    for t in ticks:
        ap = actual_price(t)
        if ap is None:
            continue
        n += 1
        pp = pred_price_fn(t.fv)
        if pp == ap:
            match += 1
            v = actual_vol(t)
            if v is not None:
                vol_ct[v] += 1
    return match, n, vol_ct


def validate_one(data_dir: Path) -> dict[str, tuple[int, int, Counter[int]]]:
    ticks, prices_path = load_ticks(data_dir)
    n = len(ticks)
    print("=" * 70)
    print("  VALIDATION: osmium_wall_bot (round(FV) ± 10 / ± 11)")
    print(f"  data_dir={data_dir.resolve()}")
    print(f"  prices={prices_path.name}  timesteps={n}")
    print("=" * 70)

    legs = [
        ("bid -10", lambda t: t.bid_m10, lambda t: t.bid_m10_vol, lambda fv: wall_prices(fv).bid_m10),
        ("bid -11", lambda t: t.bid_m11, lambda t: t.bid_m11_vol, lambda fv: wall_prices(fv).bid_m11),
        ("ask +10", lambda t: t.ask_p10, lambda t: t.ask_p10_vol, lambda fv: wall_prices(fv).ask_p10),
        ("ask +11", lambda t: t.ask_p11, lambda t: t.ask_p11_vol, lambda fv: wall_prices(fv).ask_p11),
    ]

    out: dict[str, tuple[int, int, Counter[int]]] = {}
    print("\n  PRICE (per rung)")
    print(f"  {'leg':<10} {'match':>12} {'pct':>8}")
    for label, gp, gv, pred in legs:
        m, nv, vc = leg_report(ticks, gp, gv, pred)
        out[label] = (m, nv, vc)
        print(f"  {label:<10} {m:>5}/{nv:<5} {100 * m / max(nv, 1):>7.1f}%")

    print(f"\n  VOLUME χ² @ price match (U[20..30], df=10, crit≈18.3)")
    for label, _, _, _ in legs:
        _, _, vol_ct = out[label]
        nv = sum(vol_ct.values())
        if nv == 0:
            print(f"  {label}: n=0")
            continue
        chi = chi2_uniform(vol_ct, 20, 30, nv)
        print(f"  {label}: n={nv}  chi²={chi:.2f}  pass={chi < 18.3}")

    return out


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

    legs_keys = ["bid -10", "bid -11", "ask +10", "ask +11"]
    pooled_vol: dict[str, Counter[int]] = {k: Counter() for k in legs_keys}
    pooled_mn: dict[str, tuple[int, int]] = {k: (0, 0) for k in legs_keys}

    for d in dirs:
        per = validate_one(d)
        for lab in legs_keys:
            m, nv, vc = per[lab]
            pooled_mn[lab] = (pooled_mn[lab][0] + m, pooled_mn[lab][1] + nv)
            pooled_vol[lab].update(vc)

    if len(dirs) > 1:
        print("=" * 70)
        print("  POOLED (all sessions) — price match + merged volume χ²")
        print("=" * 70)
        for lab in legs_keys:
            m, nv = pooled_mn[lab]
            print(f"  {lab}: {m}/{nv} ({100 * m / max(nv, 1):.1f}%)")
        for lab in legs_keys:
            vc = pooled_vol[lab]
            nv = sum(vc.values())
            if nv == 0:
                continue
            chi = chi2_uniform(vc, 20, 30, nv)
            print(f"  {lab} vol χ² (merged): n={nv}  chi²={chi:.2f}  pass={chi < 18.3}")


if __name__ == "__main__":
    main()
