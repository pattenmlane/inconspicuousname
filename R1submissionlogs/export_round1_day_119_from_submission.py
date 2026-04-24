#!/usr/bin/env python3
"""
Build Prosperity-style **prices** and **trades** CSVs from a website submission zip.

Reads ``{id}.log`` (JSON with ``activitiesLog`` + ``tradeHistory``). The small
``{id}.json`` export does not include ``tradeHistory``, so this script requires
the ``.log`` member inside the zip.

* **Prices:** same columns as ``ROUND1/prices_round_1_day_*.csv``; sets
  ``day`` to **119** and ``profit_and_loss`` to **0.0** (canonical price dumps).
* **Trades:** same columns as ``ROUND1/trades_round_1_day_*.csv``; drops any row
  where ``buyer == "SUBMISSION"`` or ``seller == "SUBMISSION"`` (user fills).

Usage::

  python3 R1submissionlogs/export_round1_day_119_from_submission.py \\
    --zip R1submissionlogs/273774.zip \\
    --out-dir Prosperity4Data/ROUND1
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

TARGET_DAY = 119
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
ALLOWED = {PEPPER, OSMIUM}


def find_log_bytes(zf: zipfile.ZipFile) -> bytes:
    names = [n for n in zf.namelist() if n.endswith(".log")]
    if not names:
        raise SystemExit("Zip has no .log file (need full submission export).")
    return zf.read(names[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zip",
        type=Path,
        default=Path(__file__).resolve().parent / "273774.zip",
        help="Submission zip containing .log",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Prosperity4Data" / "ROUND1",
        help="Directory for prices_round_1_day_119.csv and trades_round_1_day_119.csv",
    )
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as zf:
        raw = find_log_bytes(zf)
    data = json.loads(raw.decode("utf-8", errors="strict"))

    activities = data.get("activitiesLog")
    if not isinstance(activities, str) or not activities.strip():
        raise SystemExit("Missing activitiesLog")

    # --- prices: rewrite day column and zero PnL column ---
    r = csv.reader(io.StringIO(activities), delimiter=";")
    rows_out: list[list[str]] = []
    header = next(r)
    idx_day = header.index("day")
    idx_pnl = header.index("profit_and_loss")
    rows_out.append(header)
    for parts in r:
        if not parts or len(parts) < len(header):
            continue
        if parts[header.index("product")] not in ALLOWED:
            continue
        row = list(parts)
        while len(row) < len(header):
            row.append("")
        row[idx_day] = str(TARGET_DAY)
        row[idx_pnl] = "0.0"
        rows_out.append(row)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    price_path = out_dir / f"prices_round_1_day_{TARGET_DAY}.csv"
    with price_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerows(rows_out)

    # --- trades: exclude user (SUBMISSION) legs ---
    th = data.get("tradeHistory")
    if not isinstance(th, list):
        raise SystemExit("Missing tradeHistory in .log JSON")

    def is_user_trade(t: dict) -> bool:
        return t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"

    trades_rows: list[dict] = []
    for t in th:
        if is_user_trade(t):
            continue
        sym = t.get("symbol")
        if sym not in ALLOWED:
            continue
        trades_rows.append(t)
    trades_rows.sort(key=lambda x: (int(x["timestamp"]), x["symbol"], float(x["price"])))

    trade_path = out_dir / f"trades_round_1_day_{TARGET_DAY}.csv"
    trade_header = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]
    with trade_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";", lineterminator="\n")
        w.writerow(trade_header)
        for t in trades_rows:
            w.writerow(
                [
                    int(t["timestamp"]),
                    t.get("buyer") or "",
                    t.get("seller") or "",
                    t["symbol"],
                    t.get("currency") or "XIRECS",
                    float(t["price"]),
                    int(t["quantity"]),
                ]
            )

    print(f"Wrote {price_path} ({len(rows_out) - 1} data rows)")
    print(f"Wrote {trade_path} ({len(trades_rows)} trades, user SUBMISSION trades removed)")


if __name__ == "__main__":
    main()
