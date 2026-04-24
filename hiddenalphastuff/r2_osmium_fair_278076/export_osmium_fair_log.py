#!/usr/bin/env python3
"""
From a Prosperity website export (.log JSON: activitiesLog + tradeHistory),
emit osmium-only prices/trades CSVs (Prosperity schema) and true server FV.

Fair value (long 1 unit, no further trades):
  true_fv(t) = profit_and_loss(t) + buy_price
where buy_price is the first SUBMISSION buy of ASH_COATED_OSMIUM in tradeHistory.

Usage:
  python3 export_osmium_fair_log.py \\
    --log "/path/to/278076.log" \\
    --out-dir "/path/to/r2_osmium_fair_278076"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SYMBOL = "ASH_COATED_OSMIUM"
ROUND_N = 2

PRICES_HEADER = [
    "day",
    "timestamp",
    "product",
    "bid_price_1",
    "bid_volume_1",
    "bid_price_2",
    "bid_volume_2",
    "bid_price_3",
    "bid_volume_3",
    "ask_price_1",
    "ask_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "ask_price_3",
    "ask_volume_3",
    "mid_price",
    "profit_and_loss",
]

TRADES_HEADER = [
    "timestamp",
    "buyer",
    "seller",
    "symbol",
    "currency",
    "price",
    "quantity",
]


def infer_buy_price(trade_history: list) -> float:
    for t in trade_history:
        if t.get("symbol") != SYMBOL:
            continue
        if t.get("buyer") == "SUBMISSION" and int(t.get("quantity", 0)) >= 1:
            return float(t["price"])
    raise ValueError(
        f"No SUBMISSION buy for {SYMBOL} found in tradeHistory "
        "(need one-share fair probe log)."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True, help="Path to .log (JSON)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: parent of this script)",
    )
    args = p.parse_args()

    log_path = args.log.expanduser().resolve()
    out_dir = (args.out_dir or Path(__file__).resolve().parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(log_path.read_text(encoding="utf-8"))
    activities = (data.get("activitiesLog") or "").strip()
    if not activities:
        print("activitiesLog missing or empty", file=sys.stderr)
        sys.exit(1)

    lines = activities.split("\n")
    header = lines[0].split(";")
    if header != PRICES_HEADER:
        print("Unexpected activitiesLog header:", header, file=sys.stderr)
        sys.exit(1)

    th = data.get("tradeHistory") or []
    if not isinstance(th, list):
        print("tradeHistory missing", file=sys.stderr)
        sys.exit(1)

    buy_price = infer_buy_price(th)

    price_rows_osm: list[list[str]] = []
    fv_rows: list[dict[str, str | float]] = []

    for raw in lines[1:]:
        cols = raw.split(";")
        if len(cols) != len(PRICES_HEADER):
            raise ValueError(f"Bad row length {len(cols)}")
        if cols[2] != SYMBOL:
            continue
        price_rows_osm.append(cols)
        pnl = float(cols[16])
        mid = float(cols[15])
        ts = int(cols[1])
        day = cols[0]
        true_fv = pnl + buy_price
        fv_rows.append(
            {
                "day": day,
                "timestamp": ts,
                "mid_price": mid,
                "profit_and_loss": pnl,
                "buy_price": buy_price,
                "true_fv": true_fv,
            }
        )

    trade_rows = []
    for t in sorted(th, key=lambda x: (int(x.get("timestamp", 0)), str(x.get("symbol", "")))):
        if t.get("symbol") != SYMBOL:
            continue
        trade_rows.append(
            [
                str(int(t.get("timestamp", 0))),
                str(t.get("buyer", "")),
                str(t.get("seller", "")),
                str(t.get("symbol", "")),
                str(t.get("currency", "")),
                str(float(t.get("price", 0.0))),
                str(int(t.get("quantity", 0))),
            ]
        )

    day_token = price_rows_osm[0][0] if price_rows_osm else "1"
    prices_path = out_dir / f"prices_round_{ROUND_N}_day_{day_token}.csv"
    trades_path = out_dir / f"trades_round_{ROUND_N}_day_{day_token}.csv"
    fv_path = out_dir / "osmium_true_fv.csv"

    with prices_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(PRICES_HEADER)
        w.writerows(price_rows_osm)

    with trades_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(TRADES_HEADER)
        w.writerows(trade_rows)

    fv_fieldnames = [
        "day",
        "timestamp",
        "mid_price",
        "profit_and_loss",
        "buy_price",
        "true_fv",
    ]
    with fv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fv_fieldnames, delimiter=";")
        w.writeheader()
        for row in fv_rows:
            w.writerow({k: row[k] for k in fv_fieldnames})

    meta_path = out_dir / "export_meta.txt"
    meta_path.write_text(
        f"log={log_path}\n"
        f"symbol={SYMBOL}\n"
        f"buy_price={buy_price}\n"
        f"price_rows={len(price_rows_osm)}\n"
        f"trade_rows={len(trade_rows)}\n",
        encoding="utf-8",
    )

    print(f"buy_price (from SUBMISSION fill): {buy_price}")
    print(f"Wrote {prices_path} ({len(price_rows_osm)} rows)")
    print(f"Wrote {trades_path} ({len(trade_rows)} rows)")
    print(f"Wrote {fv_path} ({len(fv_rows)} rows)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
