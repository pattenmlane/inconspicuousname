#!/usr/bin/env python3
"""
Build prices + trades CSVs from a Prosperity submission artifact zip (.log + .json).

Example:
  python3 logtodata.py --zip "day 29 logs/278346.zip" --round 2 --day 29
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

EXPECTED_PRICES_HEADER = [
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

EXPECTED_TRADES_HEADER = [
    "timestamp",
    "buyer",
    "seller",
    "symbol",
    "currency",
    "price",
    "quantity",
]


def load_zip_member_json(zip_path: Path, member_name: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name) as f:
            return json.loads(f.read().decode("utf-8"))


def infer_stem(zip_path: Path, zf: zipfile.ZipFile) -> str:
    logs = [n for n in zf.namelist() if n.endswith(".log") and not n.startswith("__")]
    if len(logs) != 1:
        raise ValueError(f"Expected exactly one .log in zip, got: {logs}")
    return Path(logs[0]).stem


def choose_activities(json_obj: dict[str, Any], log_obj: dict[str, Any]) -> str:
    a_json = json_obj.get("activitiesLog", "") or ""
    a_log = log_obj.get("activitiesLog", "") or ""
    activities = a_json or a_log
    if not activities:
        raise ValueError("No activitiesLog found in zip artifacts")
    if a_json and a_log and a_json != a_log:
        print(
            "Warning: activitiesLog differs between .json and .log; using .log",
            file=sys.stderr,
        )
        activities = a_log
    return activities


def validate_prices_header(header: list[str]) -> None:
    if header != EXPECTED_PRICES_HEADER:
        raise ValueError(
            "Unexpected prices header (expected Prosperity activitiesLog schema):\n"
            f"  got: {header}"
        )


def build_prices(activities: str, day_override: str | None) -> tuple[list[list[str]], list[str]]:
    lines = [ln for ln in activities.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("activitiesLog is empty")
    header = lines[0].split(";")
    validate_prices_header(header)

    rows: list[list[str]] = []
    for raw in lines[1:]:
        cols = raw.split(";")
        if len(cols) != len(header):
            raise ValueError(f"Malformed prices row with {len(cols)} fields")
        if day_override is not None:
            cols[0] = day_override
        rows.append(cols)
    return rows, header


def build_trades(log_obj: dict[str, Any], exclude_submission_trades: bool = False) -> list[list[str]]:
    trade_history = log_obj.get("tradeHistory", [])
    if not isinstance(trade_history, list):
        raise ValueError("tradeHistory missing or malformed in .log payload")

    trade_history = sorted(
        trade_history,
        key=lambda t: (int(t.get("timestamp", 0)), str(t.get("symbol", ""))),
    )

    rows: list[list[str]] = []
    for t in trade_history:
        if exclude_submission_trades and (
            t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"
        ):
            continue
        rows.append(
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
    return rows


def default_out_paths(
    zip_path: Path, stem: str, out_dir: Path, round_n: int, day_override: str | None
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if day_override is not None:
        suffix = f"round_{round_n}_day_{day_override}_{stem}"
    else:
        suffix = f"round_{round_n}_{stem}"
    return out_dir / f"prices_{suffix}.csv", out_dir / f"trades_{suffix}.csv"


def main() -> None:
    p = argparse.ArgumentParser(description="Export prices/trades CSV from submission zip.")
    p.add_argument(
        "--zip",
        type=Path,
        required=True,
        help="Path to submission .zip (contains <id>.log and <id>.json)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {HERE}/logtodata_out)",
    )
    p.add_argument("--round", type=int, default=2, help="Round number for output filename")
    p.add_argument(
        "--day",
        type=str,
        default=None,
        help='If set, replace the "day" column in every prices row with this value',
    )
    p.add_argument("--stem", type=str, default=None, help="Override log/json stem (default: infer)")
    p.add_argument(
        "--exclude-submission-trades",
        action="store_true",
        help='Omit trades where buyer or seller is "SUBMISSION" (participant fills) from the export',
    )
    args = p.parse_args()

    zip_path = args.zip.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)

    out_dir = (args.out_dir or (HERE / "logtodata_out")).expanduser().resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        stem = args.stem or infer_stem(zip_path, zf)
        names = set(zf.namelist())
        log_name = f"{stem}.log"
        json_name = f"{stem}.json"
        if log_name not in names or json_name not in names:
            raise ValueError(f"Zip must contain {log_name} and {json_name}; have: {sorted(names)}")

    json_obj = load_zip_member_json(zip_path, json_name)
    log_obj = load_zip_member_json(zip_path, log_name)
    activities = choose_activities(json_obj, log_obj)

    day_override = args.day
    price_rows, header = build_prices(activities, day_override)
    trade_rows = build_trades(log_obj, exclude_submission_trades=args.exclude_submission_trades)

    out_prices, out_trades = default_out_paths(zip_path, stem, out_dir, args.round, day_override)

    with out_prices.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(price_rows)

    with out_trades.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(EXPECTED_TRADES_HEADER)
        w.writerows(trade_rows)

    print(f"Wrote {out_prices} ({len(price_rows)} rows)")
    print(f"Wrote {out_trades} ({len(trade_rows)} rows)")


if __name__ == "__main__":
    main()
