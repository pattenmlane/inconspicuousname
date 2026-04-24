#!/usr/bin/env python3
"""
Prosperity 4 — **ASH_COATED_OSMIUM** market trades only (``trades_round_1_day_*.csv``).

Hedgehogs-style *hypothesis*: some bot repeatedly prints size near **evolving daily
extrema** (Olivia on Ink: 15 lot at running low / high). P4 logs here have **empty
buyer/seller**, so we **cannot** name a trader — we only test **price × size ×
timing vs running min/max of prior trade prices** that day.

For each trade (chronological per day):
  * ``prior_min`` / ``prior_max`` = min/max **trade price** over **strictly earlier**
    trades that day.
  * ``at_low``  = prior_min is not None and ``price <= prior_min + tol``
  * ``at_high`` = prior_max is not None and ``price >= prior_max - tol``
  * ``new_low`` / ``new_high`` = first print strictly below prior min / above prior max

Reports quantity distributions, enrichment vs baseline, and overlap of ``at_low`` /
``at_high`` with modal sizes.

Usage:
  python3 Prosperity4Data/analyze_osmium_trade_extrema_insider_probe.py
  python3 Prosperity4Data/analyze_osmium_trade_extrema_insider_probe.py --tol 2
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRADES_DIR = REPO / "Prosperity4Data" / "ROUND1"
PRODUCT = "ASH_COATED_OSMIUM"
ROUND = 1


def discover_trade_days() -> list[int]:
    days: list[int] = []
    for p in TRADES_DIR.glob(f"trades_round_{ROUND}_day_*.csv"):
        m = re.search(r"day_(-?\d+)\.csv$", p.name)
        if m:
            days.append(int(m.group(1)))
    return sorted(days)


def load_osmium_trades(day: int) -> pd.DataFrame:
    path = TRADES_DIR / f"trades_round_{ROUND}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["symbol"] == PRODUCT].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def enrich_flags(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Adds at_low, at_high, new_low, new_high per row (causal / trade-time)."""
    prices: list[float] = []
    at_low: list[bool] = []
    at_high: list[bool] = []
    new_low: list[bool] = []
    new_high: list[bool] = []
    for _, row in df.iterrows():
        px = float(row["price"])
        q = int(row["quantity"])
        if not prices:
            at_low.append(False)
            at_high.append(False)
            new_low.append(False)
            new_high.append(False)
            prices.append(px)
            continue
        pmin, pmax = min(prices), max(prices)
        at_low.append(px <= pmin + tol)
        at_high.append(px >= pmax - tol)
        new_low.append(px < pmin)
        new_high.append(px > pmax)
        prices.append(px)
    out = df.copy()
    out["at_low"] = at_low
    out["at_high"] = at_high
    out["new_low"] = new_low
    out["new_high"] = new_high
    return out


def qty_dist(series: pd.Series) -> Counter:
    return Counter(int(x) for x in series)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=1.0, help="Ticks from prior extrema (price units).")
    args = p.parse_args()

    days = discover_trade_days()
    all_rows: list[pd.DataFrame] = []
    for d in days:
        t = load_osmium_trades(d)
        if t.empty:
            continue
        t["day"] = d
        t = enrich_flags(t, args.tol)
        all_rows.append(t)
    if not all_rows:
        raise SystemExit("No osmium trades found.")

    df = pd.concat(all_rows, ignore_index=True)
    n = len(df)
    n_low = int(df["at_low"].sum())
    n_high = int(df["at_high"].sum())
    n_nl = int(df["new_low"].sum())
    n_nh = int(df["new_high"].sum())

    print(f"{PRODUCT} — trade-time proximity to **prior** daily trade-price extrema")
    print(f"Days: {days}  trades: {n}  tol={args.tol} (price units)")
    b_non = ~(df["at_low"] | df["at_high"])
    base = df.loc[b_non, "quantity"]
    print()
    print("Coverage (first trade per day has no 'prior' range — flags start from 2nd trade):")
    print(f"  at_low:  {n_low:5d}  ({100*n_low/n:.2f}%)")
    print(f"  at_high: {n_high:5d}  ({100*n_high/n:.2f}%)")
    print(f"  new_low (strict new min):  {n_nl:5d}")
    print(f"  new_high (strict new max): {n_nh:5d}")
    print()

    # Enrichment: P(qty=k | at_low) vs P(qty=k | baseline)
    def report_slice(name: str, mask: pd.Series) -> None:
        sub = df.loc[mask, "quantity"]
        if len(sub) < 30:
            print(f"{name}: only {len(sub)} trades — skip detail")
            return
        c = qty_dist(sub)
        tot = len(sub)
        top = c.most_common(8)
        print(f"{name} (n={tot}) — top quantities:")
        for q, cnt in top:
            print(f"    qty {q:3d}: {cnt:5d}  ({100*cnt/tot:.1f}%)")
        mode_q, mode_n = top[0]
        base_tot = len(base)
        if base_tot > 0:
            base_mode_n = int((base == mode_q).sum())
            p_slice = mode_n / tot
            p_base = base_mode_n / base_tot
            enrich = p_slice / p_base if p_base > 0 else float("inf")
            print(f"    mode qty={mode_q}: P(slice)={p_slice:.3f} vs P(not low∪high)={p_base:.3f}  enrichment×{enrich:.2f}")
        print()

    print("--- Quantity patterns ---")
    report_slice("at_low (near prior min)", df["at_low"])
    report_slice("at_high (near prior max)", df["at_high"])
    print("Baseline: not (at_low or at_high)")
    bc = qty_dist(base)
    bt = len(base)
    if bt > 0:
        for q, cnt in bc.most_common(8):
            print(f"    qty {q:3d}: {cnt:5d}  ({100*cnt/bt:.1f}%)")
    print()

    # Same-timestamp bursts: multiple trades same ts
    dup_ts = df.groupby(["day", "timestamp"]).size()
    multi = int((dup_ts > 1).sum())
    print(f"Unique (day,timestamp) with >1 osmium trade: {multi}")
    print()
    print(
        "Interpretation: **empty buyer/seller** in these CSVs — you cannot prove *who* "
        "prints at extrema. Strong Olivia-style signal would show **one dominant qty** "
        "at lows vs highs with **large enrichment** over baseline and tight price pinning."
    )


if __name__ == "__main__":
    main()
