#!/usr/bin/env python3
"""
Enrich Round 1 **day 19** CSV with **internal fair** from a **+1 lot probe**.

Definition (what you described)
---------------------------------
You buy **1** osmium at execution price **E**. The log gives **cumulative** mark
``profit_and_loss`` each tick (same units as price for this purpose). Then for
every timestamp:

    internal_fair(t) = E + profit_and_loss(t)

So if ``E = 10_000`` and at ``t=1`` PnL is ``+4``, internal fair is ``10_004``;
if at ``t=2`` PnL is ``-2``, internal fair is ``9_998``. No wall regression, no
inversion — only **fill price** + **engine PnL**.

PnL source
----------
* Prefer ``profit_and_loss`` already in the day CSV (website export with marks).
* If that column is all zeros in your repo copy, pass ``--pnl-log`` to merge
  ``profit_and_loss`` from a JSON ``.log`` ``activitiesLog`` (same
  ``timestamp`` / ``product`` keys only — we do **not** reinterpret PnL).

Also adds book mids (``micro_mid``, ``wall_mid``, ``popular_mid``) for analysis
and ``internal_minus_micro`` = internal fair minus micro mid.

Usage::

  python3 Prosperity4Data/enrich_round1_day19_internal_fair.py --entry-price 10011
  python3 Prosperity4Data/enrich_round1_day19_internal_fair.py --entry-from-log INK_INFO/248329.log
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROUND = 1
PRODUCT_OSMIUM = "ASH_COATED_OSMIUM"


def _levels(row: pd.Series, side: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for i in range(1, 4):
        p = row.get(f"{side}_price_{i}")
        v = row.get(f"{side}_volume_{i}")
        if pd.isna(p) or pd.isna(v) or float(v) <= 0:
            continue
        out.append((float(p), float(v)))
    return out


def micro_mid_row(row: pd.Series) -> float:
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    return (max(p for p, _ in b) + min(p for p, _ in a)) / 2.0


def wall_mid_row(row: pd.Series) -> float:
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    return (min(p for p, _ in b) + max(p for p, _ in a)) / 2.0


def popular_mid_row(row: pd.Series) -> float:
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    pop_b = max(b, key=lambda t: t[1])[0]
    pop_a = max(a, key=lambda t: t[1])[0]
    return (pop_b + pop_a) / 2.0


def entry_price_from_log(log_path: Path, product: str = PRODUCT_OSMIUM) -> float:
    obj = json.loads(log_path.read_text(encoding="utf-8"))
    th = obj.get("tradeHistory")
    if not isinstance(th, list):
        raise SystemExit("log JSON missing tradeHistory list")
    for tr in th:
        if tr.get("buyer") != "SUBMISSION":
            continue
        if tr.get("symbol") != product:
            continue
        q = int(tr.get("quantity", 0))
        if q <= 0:
            continue
        return float(tr["price"])
    raise SystemExit(f"No SUBMISSION buy found for {product} in {log_path}")


def load_pnl_from_log(log_path: Path) -> pd.Series:
    obj = json.loads(log_path.read_text(encoding="utf-8"))
    al = obj.get("activitiesLog")
    if not isinstance(al, str):
        raise SystemExit("log JSON missing activitiesLog string")
    dlog = pd.read_csv(io.StringIO(al), sep=";")
    return dlog.set_index(["timestamp", "product"])["profit_and_loss"]


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="internal_fair = entry + PnL (probe)")
    ap.add_argument(
        "--day-csv",
        type=Path,
        default=root / "ROUND1" / f"prices_round_{ROUND}_day_19.csv",
    )
    ap.add_argument(
        "--entry-price",
        type=float,
        default=None,
        help="Fill price E for the +1 osmium probe (required unless --entry-from-log).",
    )
    ap.add_argument(
        "--entry-from-log",
        type=Path,
        default=None,
        help="JSON .log: read first SUBMISSION buy price for osmium as E.",
    )
    ap.add_argument(
        "--pnl-log",
        type=Path,
        default=None,
        help="Optional: merge profit_and_loss from this .log activitiesLog when CSV PnL is empty.",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "ROUND1" / f"prices_round_{ROUND}_day_19_enriched.csv",
    )
    args = ap.parse_args()

    if not args.day_csv.is_file():
        sys.exit(f"Missing day CSV: {args.day_csv}")

    if args.entry_from_log is not None:
        if not args.entry_from_log.is_file():
            sys.exit(f"Missing --entry-from-log: {args.entry_from_log}")
        entry = entry_price_from_log(args.entry_from_log)
    elif args.entry_price is not None:
        entry = float(args.entry_price)
    else:
        sys.exit("Provide --entry-price E or --entry-from-log path.to.log")

    df = pd.read_csv(args.day_csv, sep=";")

    pnl_col = pd.to_numeric(df["profit_and_loss"], errors="coerce")
    if args.pnl_log is not None:
        if not args.pnl_log.is_file():
            sys.exit(f"Missing --pnl-log: {args.pnl_log}")
        pnl_series = load_pnl_from_log(args.pnl_log)

        def lookup(r: pd.Series) -> float:
            key = (int(r["timestamp"]), str(r["product"]))
            if key not in pnl_series.index:
                return float("nan")
            return float(pnl_series.loc[key])

        pnl_col = df.apply(lookup, axis=1)

    df["micro_mid"] = df.apply(micro_mid_row, axis=1)
    df["wall_mid"] = df.apply(wall_mid_row, axis=1)
    df["popular_mid"] = df.apply(popular_mid_row, axis=1)

    # internal_fair = E + PnL  (osmium only; probe assumption)
    internal = np.full(len(df), np.nan, dtype=float)
    osm_mask = df["product"] == PRODUCT_OSMIUM
    p_osm = pnl_col[osm_mask].to_numpy(dtype=float)
    internal[osm_mask.to_numpy()] = entry + p_osm
    df["internal_fair"] = internal
    df["probe_entry_price"] = np.where(osm_mask, entry, np.nan)
    df["internal_minus_micro"] = df["internal_fair"] - df["micro_mid"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep=";", index=False)

    print(f"Wrote {args.output}")
    print(f"Probe entry E = {entry:g}  (internal_fair = E + profit_and_loss on {PRODUCT_OSMIUM})")
    if args.pnl_log:
        print(f"PnL merged from {args.pnl_log}")
    else:
        print("PnL from CSV column profit_and_loss")


if __name__ == "__main__":
    main()
