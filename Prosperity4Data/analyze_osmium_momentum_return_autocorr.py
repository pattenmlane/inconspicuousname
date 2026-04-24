#!/usr/bin/env python3
"""
ASH_COATED_OSMIUM — **return persistence** on mid-like series (ROUND1 ``prices_round_*``).

1. **Lag autocorrelation** of first differences ``Δm[t] = m[t+1]-m[t]`` within each day,
   pooled as a weighted average of per-day correlations (by overlap length).

2. **Naive one-step “paper” PnL** (no spread/fees/inventory; sign of prior move only):
   * **MR:** position ``-sign(Δm[t])``, PnL ``pos * Δm[t+1]``
   * **MOM:** position ``+sign(Δm[t])``, same PnL

Positive mean PnL for MR ⇒ short-horizon mean reversion; for MOM ⇒ momentum.

Reuses mid construction from ``analyze_osmium_zscore_meanrev.py``.

Usage:
  python3 Prosperity4Data/analyze_osmium_momentum_return_autocorr.py
  python3 Prosperity4Data/analyze_osmium_momentum_return_autocorr.py --mid csv_mid
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_osmium_zscore_meanrev import discover_days, mid_series  # noqa: E402
from plot_osmium_micro_mid_vs_vol_mid import PRODUCT, ROUND  # noqa: E402


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / f"ROUND{ROUND}"


def lag_autocorr_per_day(m: np.ndarray, lag: int) -> float | None:
    """acf of np.diff(m) at `lag`; None if undefined."""
    d = np.diff(m.astype(np.float64))
    if len(d) <= lag + 2:
        return None
    x, y = d[:-lag], d[lag:]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def day_pnl_mr_mom(m: np.ndarray) -> tuple[float, float, int, int]:
    """
    Returns (mean_pnl_mr, mean_pnl_mom, n_used_mr, n_used_mom) on non-zero prior moves.
    Skips Δm[t]==0 for position definition.
    """
    d = np.diff(m.astype(np.float64))
    if len(d) < 3:
        return float("nan"), float("nan"), 0, 0
    prev, nxt = d[:-1], d[1:]
    mask = prev != 0.0
    prev, nxt = prev[mask], nxt[mask]
    if len(prev) == 0:
        return float("nan"), float("nan"), 0, 0
    pos_mr = -np.sign(prev)
    pos_mo = np.sign(prev)
    pnl_mr = pos_mr * nxt
    pnl_mo = pos_mo * nxt
    return float(pnl_mr.mean()), float(pnl_mo.mean()), int(len(pnl_mr)), int(len(pnl_mo))


def main() -> None:
    ap = argparse.ArgumentParser(description="Momentum vs MR from mid return autocorrelation.")
    ap.add_argument("--days", "-d", type=int, nargs="*", default=None)
    ap.add_argument(
        "--mid",
        choices=("vol", "jmerle", "wall", "micro", "csv_mid"),
        default="vol",
        help="Mid series (default: vol = popular mid).",
    )
    args = ap.parse_args()

    root = _data_dir()
    days = sorted(args.days) if args.days else discover_days(root)
    if not days:
        raise SystemExit("No prices CSVs.")

    df = mid_series(root, days, args.mid)
    print(f"{PRODUCT} — return persistence & naive tick MR vs MOM  (mid={args.mid!r})")
    print(f"Days: {days}  mid rows: {len(df)}")
    print()

    lags = [1, 2, 3, 5, 10, 20]
    print("--- Lag autocorrelation of Δm (within-day; mean of per-day corrs) ---")
    for lag in lags:
        acs: list[float] = []
        for _, g in df.groupby("day", sort=False):
            m = g["m"].to_numpy()
            c = lag_autocorr_per_day(m, lag)
            if c is not None:
                acs.append(c)
        if not acs:
            print(f"  lag {lag:2d}: (no valid days)")
        else:
            mu = float(np.mean(acs))
            sd = float(np.std(acs))
            print(f"  lag {lag:2d}:  mean acf = {mu:+.4f}  (std across days {sd:.4f}, n_days={len(acs)})")
    print("  (+ => momentum at that tick gap, − => mean reversion)")
    print()

    print("--- Naive one-step paper PnL: pos * next Δm (only |prior Δm|>0) ---")
    mrs, mos, ns = [], [], []
    for _, g in df.groupby("day", sort=False):
        mr, mo, n, _ = day_pnl_mr_mom(g["m"].to_numpy())
        if n > 0 and not np.isnan(mr):
            mrs.append(mr)
            mos.append(mo)
            ns.append(n)
    if mrs:
        print(
            f"  MR  (-sign prior):  mean daily mean-PnL = {float(np.mean(mrs)):+.5f}  "
            f"(std {float(np.std(mrs)):.5f} across {len(mrs)} days)  avg n/day ≈ {float(np.mean(ns)):.0f}"
        )
        print(
            f"  MOM (+sign prior):  mean daily mean-PnL = {float(np.mean(mos)):+.5f}  "
            f"(std {float(np.std(mos)):.5f} across {len(mos)} days)"
        )
        if float(np.mean(mrs)) > float(np.mean(mos)):
            print("  → MR rule beats MOM on this crude one-tick score (before costs).")
        elif float(np.mean(mrs)) < float(np.mean(mos)):
            print("  → MOM rule beats MR on this crude one-tick score (before costs).")
        else:
            print("  → Roughly tied.")
    print()

    print("--- Z-score fade check (same as zscore_meanrev quick view) ---")
    from analyze_osmium_zscore_meanrev import add_forward_by_day, smoothed_z  # noqa: E402

    d2 = df.copy()
    d2["sig"] = d2.groupby("day", sort=False)["m"].transform(lambda s: smoothed_z(s, 30, 20))
    add_forward_by_day(d2, [1])
    v = d2["sig"].notna() & d2["f1"].notna()
    sub = d2.loc[v]
    if len(sub) > 50:
        c = sub["sig"].corr(sub["f1"])
        print(f"  corr(smoothed_z Wz=30 Ws=20, next Δm): {c:+.4f}  (negative => fade/MR)")
        hi = sub["sig"] > 1.0
        lo = sub["sig"] < -1.0
        print(f"  mean next Δm | sig>+1:  {sub.loc[hi, 'f1'].mean():+.5f}  (n={hi.sum()})")
        print(f"  mean next Δm | sig<-1: {sub.loc[lo, 'f1'].mean():+.5f}  (n={lo.sum()})")
    print()
    print("Caveat: one-tick sign rules ignore spread and position; use as directional evidence only.")


if __name__ == "__main__":
    main()
