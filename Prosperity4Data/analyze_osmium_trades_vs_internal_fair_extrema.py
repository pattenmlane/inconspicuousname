#!/usr/bin/env python3
"""
**ASH_COATED_OSMIUM** market trades vs **true internal fair** from the +1 probe:

    internal_fair(t) = E + profit_and_loss(t)

using the same merge as ``plot_internal_fair_mid_wall_day19.py`` / ``enrich_round1_day19_internal_fair.py``.

For a chosen day (default **Round 1 day 19**):

1. Build the fair series on the price tape (osmium rows only).
2. ``fair_day_max`` / ``fair_day_min`` = finite extrema over the day.
3. Join each **market** trade (``trades_round_*_day_*.csv``) on ``timestamp`` — for
   day 19 in this repo every trade ts matches a price-row ts.

Flags (``--tol`` in **price ticks**):

* **near_day_high** — ``abs(price - fair_day_max) <= tol`` (print is pinned to the
  day's **maximum** mark).
* **near_day_low** — ``abs(price - fair_day_min) <= tol`` (pinned to day's **minimum** mark).
* **fair_at_peak** — ``abs(internal_fair_at_ts - fair_day_max) <= tol`` (mark itself is at the high).
* **fair_at_trough** — same for the low.

Buyer/seller are empty in these exports — this does not identify *who* traded; it only
surfaces **when** public prints sit on the probe's daily fair envelope.

Usage::

  cd ProsperityRepo
  python3 Prosperity4Data/analyze_osmium_trades_vs_internal_fair_extrema.py \\
    --pnl-log INK_INFO/248329.log --entry-from-log INK_INFO/248329.log

  python3 Prosperity4Data/analyze_osmium_trades_vs_internal_fair_extrema.py \\
    --pnl-log INK_INFO/248329.log --entry-from-log INK_INFO/248329.log --tol 1.5 \\
    --csv-out Prosperity4Data/ROUND1/osmium_trades_near_internal_fair_extrema_r1d19.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from enrich_round1_day19_internal_fair import entry_price_from_log, load_pnl_from_log

REPO_DATA = Path(__file__).resolve().parent
ROUND1 = REPO_DATA / "ROUND1"
PRODUCT = "ASH_COATED_OSMIUM"
ROUND = 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Market trades vs probe internal_fair day high/low.")
    ap.add_argument("--day", type=int, default=19, help="Day number (Round 1).")
    ap.add_argument(
        "--day-csv",
        type=Path,
        default=None,
        help="Prices CSV (default: ROUND1/prices_round_1_day_<day>.csv).",
    )
    ap.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
        help="Trades CSV (default: ROUND1/trades_round_1_day_<day>.csv).",
    )
    ap.add_argument("--pnl-log", type=Path, required=True)
    ap.add_argument("--entry-from-log", type=Path, default=None)
    ap.add_argument("--entry-price", type=float, default=None)
    ap.add_argument("--tol", type=float, default=2.0, help="Half-width in price units for 'near' extrema.")
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Write merged trade table with flags to this path.",
    )
    args = ap.parse_args()

    day = args.day
    day_csv = args.day_csv or (ROUND1 / f"prices_round_{ROUND}_day_{day}.csv")
    trades_csv = args.trades_csv or (ROUND1 / f"trades_round_{ROUND}_day_{day}.csv")

    if not day_csv.is_file():
        sys.exit(f"Missing {day_csv}")
    if not trades_csv.is_file():
        sys.exit(f"Missing {trades_csv}")
    if not args.pnl_log.is_file():
        sys.exit(f"Missing --pnl-log {args.pnl_log}")

    if args.entry_from_log is not None:
        if not args.entry_from_log.is_file():
            sys.exit(f"Missing --entry-from-log {args.entry_from_log}")
        entry = entry_price_from_log(args.entry_from_log)
    elif args.entry_price is not None:
        entry = float(args.entry_price)
    else:
        sys.exit("Provide --entry-from-log or --entry-price")

    pnl_ser = load_pnl_from_log(args.pnl_log)
    pnl_df = pnl_ser.reset_index()
    pnl_df = pnl_df.rename(columns={"profit_and_loss": "pnl"})

    prices = pd.read_csv(day_csv, sep=";")
    osm = prices.loc[prices["product"] == PRODUCT, ["timestamp", "product"]].copy()
    osm = osm.merge(pnl_df, on=["timestamp", "product"], how="left")
    osm["internal_fair"] = entry + pd.to_numeric(osm["pnl"], errors="coerce")

    fair_arr = osm["internal_fair"].to_numpy(dtype=float)
    finite = fair_arr[np.isfinite(fair_arr)]
    if finite.size == 0:
        sys.exit("No finite internal_fair values (check PnL merge).")
    fair_max = float(np.max(finite))
    fair_min = float(np.min(finite))
    ts_at_max = osm.loc[np.isclose(osm["internal_fair"], fair_max, rtol=0.0, atol=1e-9), "timestamp"]
    ts_at_min = osm.loc[np.isclose(osm["internal_fair"], fair_min, rtol=0.0, atol=1e-9), "timestamp"]

    trades = pd.read_csv(trades_csv, sep=";")
    tr = trades.loc[trades["symbol"] == PRODUCT].copy()
    tr = tr.sort_values("timestamp").reset_index(drop=True)
    if tr.empty:
        sys.exit(f"No {PRODUCT} trades in {trades_csv}")

    fair_by_ts = osm.set_index("timestamp")["internal_fair"]
    tr["internal_fair_at_ts"] = tr["timestamp"].map(fair_by_ts)
    missing = tr["internal_fair_at_ts"].isna().sum()
    if missing:
        print(f"WARNING: {missing} trades have no matching price-row timestamp (no fair); dropping for flags.")
        tr = tr.loc[tr["internal_fair_at_ts"].notna()].copy()

    tol = float(args.tol)
    price = tr["price"].astype(float)
    f_ts = tr["internal_fair_at_ts"].astype(float)

    tr["dist_to_fair_day_high"] = fair_max - price
    tr["dist_to_fair_day_low"] = price - fair_min
    tr["price_minus_fair_at_ts"] = price - f_ts
    tr["near_day_high"] = (price - fair_max).abs() <= tol
    tr["near_day_low"] = (price - fair_min).abs() <= tol
    tr["fair_at_peak"] = (f_ts - fair_max).abs() <= tol
    tr["fair_at_trough"] = (f_ts - fair_min).abs() <= tol
    tr["pin_peak"] = tr["near_day_high"] & tr["fair_at_peak"]
    tr["pin_trough"] = tr["near_day_low"] & tr["fair_at_trough"]

    n = len(tr)
    n_nh = int(tr["near_day_high"].sum())
    n_nl = int(tr["near_day_low"].sum())
    n_pp = int(tr["pin_peak"].sum())
    n_pt = int(tr["pin_trough"].sum())

    print(f"{PRODUCT} — R{ROUND} day {day}  trades={n}  tol={tol}")
    print(f"Probe entry E = {entry:g}  internal_fair = E + PnL from {args.pnl_log.name}")
    span = fair_max - fair_min
    print(f"fair_day_min = {fair_min:g}  fair_day_max = {fair_max:g}  (finite ticks on price tape)")
    print(f"Day range (max - min) = {span:g} price units — wide span ⇒ fewer random hits near envelope.")
    print(f"Timestamps where fair == max: {len(ts_at_max)} rows  (show first 5): {ts_at_max.head(5).tolist()}")
    print(f"Timestamps where fair == min: {len(ts_at_min)} rows  (show first 5): {ts_at_min.head(5).tolist()}")
    print()
    print("Flags (price near **day** envelope of fair; fair_at_* = mark at trade ts near envelope):")
    print(f"  near_day_high   (|price - max| <= tol): {n_nh:3d}  ({100 * n_nh / n:.1f}%)")
    print(f"  near_day_low    (|price - min| <= tol): {n_nl:3d}  ({100 * n_nl / n:.1f}%)")
    print(f"  pin_peak  (near high AND mark at peak): {n_pp:3d}")
    print(f"  pin_trough (near low AND mark at trough): {n_pt:3d}")
    print()

    cols = [
        "timestamp",
        "price",
        "quantity",
        "internal_fair_at_ts",
        "price_minus_fair_at_ts",
        "dist_to_fair_day_high",
        "dist_to_fair_day_low",
        "near_day_high",
        "near_day_low",
        "fair_at_peak",
        "fair_at_trough",
        "pin_peak",
        "pin_trough",
    ]
    interesting = tr.loc[tr["near_day_high"] | tr["near_day_low"]].sort_values("timestamp")
    if interesting.empty:
        print("No trades within tol of day fair max or min.")
    else:
        print(f"Trades within tol of day fair high or low (n={len(interesting)}):")
        with pd.option_context("display.max_rows", 80, "display.width", 200, "display.float_format", lambda x: f"{x:.4g}"):
            print(interesting[cols].to_string(index=False))
        print()

    # Closest prints to envelope (even if outside tol) — top few each side
    show = min(8, n)
    print(f"Closest trade prices to fair_day_max (smallest dist_to_fair_day_high, top {show}):")
    top_hi = tr.nsmallest(show, "dist_to_fair_day_high")[cols]
    with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:.4g}"):
        print(top_hi.to_string(index=False))
    print()
    print(f"Closest trade prices to fair_day_min (smallest dist_to_fair_day_low, top {show}):")
    top_lo = tr.nsmallest(show, "dist_to_fair_day_low")[cols]
    with pd.option_context("display.width", 200, "display.float_format", lambda x: f"{x:.4g}"):
        print(top_lo.to_string(index=False))

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        tr_out = tr.assign(
            fair_day_max=fair_max,
            fair_day_min=fair_min,
            probe_entry=entry,
            tol=tol,
        )
        tr_out.to_csv(args.csv_out, sep=";", index=False)
        print()
        print(f"Wrote {args.csv_out}")


if __name__ == "__main__":
    main()
