#!/usr/bin/env python3
"""
Merge **probe** session logs into one CSV of **internal true fair** vs time.

Fair formula (same as Round 1 day-19 / probe writeups)
-----------------------------------------------
You buy **1** lot at execution price **E** (first ``tradeHistory`` row with
``buyer == "SUBMISSION"`` for that symbol). The activity row carries cumulative
mark ``profit_and_loss`` for that symbol:

    internal_fair(t) = E + profit_and_loss(t)

Use **pepper** probe log rows for ``INTARIAN_PEPPER_ROOT`` and **osmium** probe
log rows for ``ASH_COATED_OSMIUM`` so each **E** matches the probe that
generated that mark.

Default inputs (under this folder)::

  round2work/pepper fair/278237.log
  round2work/osmium fair/278076.log

Output::

  round2work/pepper_osmium_internal_fair.csv

Columns: ``day``, ``timestamp``, ``pepper_internal_fair``, ``osmium_internal_fair``
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
HERE = Path(__file__).resolve().parent


def _entry_price(trade_history: list[dict], symbol: str) -> float:
    for tr in trade_history:
        if tr.get("buyer") == "SUBMISSION" and tr.get("symbol") == symbol:
            return float(tr["price"])
    raise ValueError(f"No SUBMISSION buy found for {symbol}")


def _fair_series_from_log(log_path: Path, symbol: str) -> dict[tuple[int, int], float]:
    data = json.loads(log_path.read_text(encoding="utf-8"))
    E = _entry_price(data.get("tradeHistory") or [], symbol)
    out: dict[tuple[int, int], float] = {}
    r = csv.reader(io.StringIO(data["activitiesLog"]), delimiter=";")
    header = next(r)
    idx = {name: header.index(name) for name in header}
    for row in r:
        if len(row) <= idx["profit_and_loss"]:
            continue
        if row[idx["product"]] != symbol:
            continue
        day = int(row[idx["day"]])
        ts = int(row[idx["timestamp"]])
        pnl = float(row[idx["profit_and_loss"]])
        out[(day, ts)] = E + pnl
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pepper-log",
        type=Path,
        default=HERE / "pepper fair" / "278237.log",
        help="Probe JSON .log for pepper (SUBMISSION buy on INTARIAN…)",
    )
    ap.add_argument(
        "--osmium-log",
        type=Path,
        default=HERE / "osmium fair" / "278076.log",
        help="Probe JSON .log for osmium (SUBMISSION buy on ASH…)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "pepper_osmium_internal_fair.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    pep = _fair_series_from_log(args.pepper_log, PEPPER)
    osm = _fair_series_from_log(args.osmium_log, OSMIUM)
    keys = sorted(set(pep) | set(osm))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["day", "timestamp", "pepper_internal_fair", "osmium_internal_fair"])
        for day, ts in keys:
            pv = pep.get((day, ts))
            ov = osm.get((day, ts))
            w.writerow(
                [
                    day,
                    ts,
                    f"{pv:.12g}" if pv is not None else "",
                    f"{ov:.12g}" if ov is not None else "",
                ]
            )

    print(f"Wrote {args.out} ({len(keys)} rows)")


if __name__ == "__main__":
    main()
