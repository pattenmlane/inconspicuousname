#!/usr/bin/env python3
"""
From a Prosperity website export (.log JSON: activitiesLog + tradeHistory),
emit prices/trades CSVs (Round 3 schema) and true_fv for a **single-product**
hold-1 fair probe (see trader_hold1_fair_probe.py).

  true_fv(t) = profit_and_loss(t) + buy_price

buy_price = first SUBMISSION buy price for the probe symbol in tradeHistory.

Output day column is normalized to 39 to match existing fair bundles
(prices_round_3_day_39.csv, <PRODUCT>_true_fv_day39.csv).

Usage:
  python3 export_hold1_log_round3.py --log PATH/364748.log --out-dir PATH/364748
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROUND_N = 3
LABEL_DAY = "39"

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


def infer_product_and_buy_price(trade_history: list) -> tuple[str, float]:
    for t in trade_history:
        if t.get("buyer") == "SUBMISSION" and int(t.get("quantity", 0)) >= 1:
            sym = str(t.get("symbol", ""))
            return sym, float(t["price"])
    raise ValueError(
        "No SUBMISSION buy with quantity>=1 in tradeHistory (need hold-1 probe log)."
    )


def fv_csv_name(product: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", product)
    return f"{safe}_true_fv_day{LABEL_DAY}.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    log_path = args.log.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
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

    product, buy_price = infer_product_and_buy_price(th)

    price_rows: list[list[str]] = []
    fv_rows: list[dict[str, str | float]] = []

    for raw in lines[1:]:
        cols = raw.split(";")
        if len(cols) != len(PRICES_HEADER):
            continue
        if cols[2] != product:
            continue
        out_cols = cols.copy()
        out_cols[0] = LABEL_DAY
        price_rows.append(out_cols)
        pnl = float(cols[16])
        mid = float(cols[15])
        ts = int(cols[1])
        true_fv = pnl + buy_price
        fv_rows.append(
            {
                "day": LABEL_DAY,
                "timestamp": ts,
                "mid_price": mid,
                "profit_and_loss": pnl,
                "buy_price": buy_price,
                "true_fv": true_fv,
            }
        )

    if not price_rows:
        print(f"No activitiesLog rows for product={product!r}", file=sys.stderr)
        sys.exit(1)

    prices_path = out_dir / f"prices_round_{ROUND_N}_day_{LABEL_DAY}.csv"
    trades_path = out_dir / f"trades_round_{ROUND_N}_day_{LABEL_DAY}.csv"
    fv_path = out_dir / fv_csv_name(product)

    with prices_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(PRICES_HEADER)
        w.writerows(price_rows)

    trade_rows = []
    for t in sorted(th, key=lambda x: (int(x.get("timestamp", 0)), str(x.get("symbol", "")))):
        if t.get("symbol") != product:
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

    with trades_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(TRADES_HEADER)
        w.writerows(trade_rows)

    fv_fieldnames = ["day", "timestamp", "mid_price", "profit_and_loss", "buy_price", "true_fv"]
    with fv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fv_fieldnames, delimiter=";")
        w.writeheader()
        for row in fv_rows:
            w.writerow({k: row[k] for k in fv_fieldnames})

    meta_path = out_dir / f"export_meta_day{LABEL_DAY}.txt"
    meta_path.write_text(
        f"log={log_path}\n"
        f"symbol={product}\n"
        f"buy_price={buy_price}\n"
        f"price_rows={len(price_rows)}\n"
        f"trade_rows={len(trade_rows)}\n"
        f"labeled_day={LABEL_DAY}\n"
        f"fv_csv={fv_path.name}\n",
        encoding="utf-8",
    )

    print(f"product={product} buy_price={buy_price}")
    print(f"Wrote {prices_path} ({len(price_rows)} rows)")
    print(f"Wrote {trades_path} ({len(trade_rows)} rows)")
    print(f"Wrote {fv_path} ({len(fv_rows)} rows)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
