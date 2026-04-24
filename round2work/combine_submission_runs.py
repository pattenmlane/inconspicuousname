#!/usr/bin/env python3
"""
Merge multiple submission zips (same day / replay) into denser prices + trades CSVs.

Order book: for each (day, timestamp, product), fills empty bid/ask/mid cells from any
run that has a value; on conflicting non-empty values, uses majority vote (tie: first run).

PnL: taken from the run whose row had the most non-empty book fields for that key.

Trades: union of tradeHistory rows across runs, deduped by full row tuple, sorted.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# activitiesLog has 17 columns; book fields are indices 3..15, profit_and_loss is 16.
BOOK_SLICE = slice(3, 16)
PNL_IDX = 16
N_BOOK_COLS = 16 - 3  # 13

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


def is_empty(s: str) -> bool:
    return s.strip() == ""


def load_activities_from_zip(zip_path: Path) -> tuple[str, str]:
    stem = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        log_name = f"{stem}.log"
        if log_name not in names:
            logs = [n for n in names if n.endswith(".log")]
            if len(logs) != 1:
                raise ValueError(f"{zip_path}: expected one .log, got {logs}")
            stem = Path(logs[0]).stem
            log_name = logs[0]
        data = json.loads(zf.read(log_name).decode("utf-8"))
    act = data.get("activitiesLog") or ""
    if not act:
        raise ValueError(f"{zip_path}: no activitiesLog")
    return stem, act


def load_trades_from_zip(zip_path: Path) -> list[dict[str, Any]]:
    stem = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        log_name = f"{stem}.log"
        if log_name not in names:
            logs = [n for n in names if n.endswith(".log")]
            stem = Path(logs[0]).stem
            log_name = logs[0]
        data = json.loads(zf.read(log_name).decode("utf-8"))
    th = data.get("tradeHistory", [])
    if not isinstance(th, list):
        return []
    return th


def parse_price_rows(activities: str) -> list[list[str]]:
    lines = [ln for ln in activities.splitlines() if ln.strip()]
    header = lines[0].split(";")
    if header != PRICES_HEADER:
        raise ValueError(f"Unexpected header: {header[:5]}...")
    rows: list[list[str]] = []
    for raw in lines[1:]:
        cols = raw.split(";")
        if len(cols) != len(header):
            raise ValueError(f"Bad row len {len(cols)}")
        rows.append(cols)
    return rows


def row_key(row: list[str]) -> tuple[str, str, str]:
    return (row[0], row[1], row[2])


def book_completeness(row: list[str]) -> int:
    return sum(1 for c in row[BOOK_SLICE] if not is_empty(c))


def merge_cell(values: list[str]) -> tuple[str, bool]:
    """Majority vote; multimodal tie breaks to earliest run. Returns (merged_value, had_conflict)."""
    nonempty = [(i, v) for i, v in enumerate(values) if not is_empty(v)]
    if not nonempty:
        return "", False
    c = Counter(v for _, v in nonempty)
    top_ct = max(c.values())
    candidates = [v for v, ct in c.items() if ct == top_ct]
    conflict = len(candidates) > 1
    if not conflict:
        return candidates[0], False
    for _, v in nonempty:
        if v in candidates:
            return v, True
    return candidates[0], True


def merge_runs(
    run_rows: list[list[str]],
) -> tuple[list[str], int, int]:
    """
    run_rows: same key, one full row per run (same column count).
    Returns merged_row, n_conflicts (cells), n_fills (cells empty in run 0 but nonempty after merge).
    """
    merged = [""] * len(run_rows[0])
    merged[0:3] = run_rows[0][0:3]

    conflicts = 0
    fills = 0

    best_i = max(range(len(run_rows)), key=lambda i: book_completeness(run_rows[i]))
    merged[PNL_IDX] = run_rows[best_i][PNL_IDX]

    for j in range(3, 16):
        vals = [r[j] for r in run_rows]
        merged[j], cell_conflict = merge_cell(vals)
        if cell_conflict:
            conflicts += 1
        if is_empty(run_rows[0][j]) and not is_empty(merged[j]):
            fills += 1

    return merged, conflicts, fills


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "zips",
        nargs="*",
        type=Path,
        help="Zip paths (default: all day 29 logs/*.zip next to round2work)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    ap.add_argument("--day-label", type=str, default="29", help="Replace day column in output")
    args = ap.parse_args()

    if args.zips:
        zips = [p.expanduser().resolve() for p in args.zips]
    else:
        zips = sorted((HERE / "day 29 logs").glob("*.zip"))
    zips = [p for p in zips if p.is_file()]
    if not zips:
        raise SystemExit("No zip files found")

    out_dir = (args.out_dir or (HERE / "day 29 logs" / "combined")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # stem -> rows dict key -> row
    per_run: list[tuple[str, dict[tuple[str, str, str], list[str]]]] = []
    for zp in zips:
        stem, act = load_activities_from_zip(zp)
        rows = parse_price_rows(act)
        m = {row_key(r): r for r in rows}
        per_run.append((stem, m))

    keys = set()
    for _, m in per_run:
        keys |= set(m.keys())
    keys_sorted = sorted(keys, key=lambda k: (int(k[0]), int(k[1]), k[2]))

    merged_rows: list[list[str]] = []
    total_conflicts = 0
    total_fills = 0
    for key in keys_sorted:
        run_rows = []
        for stem, m in per_run:
            if key not in m:
                raise ValueError(f"Missing key {key} in {stem}")
            run_rows.append(m[key])
        merged, cf, fl = merge_runs(run_rows)
        if args.day_label:
            merged[0] = args.day_label
        merged_rows.append(merged)
        total_conflicts += cf
        total_fills += fl

    n_book_cells = len(merged_rows) * N_BOOK_COLS
    first_map = per_run[0][1]
    filled_run0 = sum(
        1 for k in keys_sorted for j in range(3, 16) if not is_empty(first_map[k][j])
    )
    filled_merged = sum(1 for r in merged_rows for j in range(3, 16) if not is_empty(r[j]))
    still_empty = 0
    for key in keys_sorted:
        for j in range(3, 16):
            if any(not is_empty(m[key][j]) for _, m in per_run):
                continue
            still_empty += 1

    prices_path = out_dir / "prices_combined_all_runs.csv"
    with prices_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(PRICES_HEADER)
        w.writerows(merged_rows)

    # trades union
    seen: set[tuple[str, ...]] = set()
    trade_rows: list[list[str]] = []
    for zp in zips:
        for t in load_trades_from_zip(zp):
            row = (
                str(int(t.get("timestamp", 0))),
                str(t.get("buyer", "")),
                str(t.get("seller", "")),
                str(t.get("symbol", "")),
                str(t.get("currency", "")),
                str(float(t.get("price", 0.0))),
                str(int(t.get("quantity", 0))),
            )
            if row in seen:
                continue
            seen.add(row)
            trade_rows.append(list(row))
    trade_rows.sort(key=lambda r: (int(r[0]), r[3], r[1], r[2]))

    trades_path = out_dir / "trades_combined_all_runs.csv"
    with trades_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(TRADES_HEADER)
        w.writerows(trade_rows)

    report_path = out_dir / "merge_report.txt"
    lines = [
        f"zips ({len(zips)}):",
        *[f"  {p.name}" for p in zips],
        "",
        f"price rows (keys): {len(merged_rows)}",
        f"book cells total (rows x {N_BOOK_COLS} cols): {n_book_cells}",
        f"non-empty book cells first zip ({per_run[0][0]}): {filled_run0} ({100 * filled_run0 / n_book_cells:.1f}%)",
        f"non-empty book cells merged: {filled_merged} ({100 * filled_merged / n_book_cells:.1f}%)",
        f"book cells empty in ALL runs (unrecoverable): {still_empty}",
        f"book cells filled vs first zip only: {total_fills}",
        f"book cells multimodal tie (earliest-run tie-break): {total_conflicts}",
        f"unique trades after union+dedupe: {len(trade_rows)}",
        "",
        f"wrote {prices_path}",
        f"wrote {trades_path}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
