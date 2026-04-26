#!/usr/bin/env python3
"""Parse a prosperity4bt JSON log: total PnL, per-symbol PnL, trade stats."""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any


def parse_activities(activities_log: str) -> tuple[float, dict[str, float], int]:
    r = csv.reader(io.StringIO(activities_log), delimiter=";")
    header = next(r, None)
    if not header or "product" not in header:
        return 0.0, {}, 0
    i_day = header.index("day")
    i_ts = header.index("timestamp")
    i_prod = header.index("product")
    i_pnl = header.index("profit_and_loss")
    max_ts = -1
    rows: list[tuple[int, int, str, float]] = []
    for row in r:
        if len(row) <= i_pnl:
            continue
        try:
            ts = int(row[i_ts])
        except ValueError:
            continue
        if ts > max_ts:
            max_ts = ts
        rows.append((int(row[i_day]), ts, row[i_prod], float(row[i_pnl])))

    pnl_by_symbol: dict[str, float] = {}
    for d, ts, sym, pnl in rows:
        if ts == max_ts:
            pnl_by_symbol[sym] = pnl
    total = sum(pnl_by_symbol.values())
    return total, pnl_by_symbol, len(rows)


def parse_trades(trade_history: list[dict[str, Any]]) -> dict[str, Any]:
    by_sym: dict[str, list[int]] = collections.defaultdict(list)
    for t in trade_history:
        sym = t.get("symbol", "")
        q = int(t.get("quantity", 0) or 0)
        by_sym[sym].append(q)
    fill_count = {k: len(v) for k, v in by_sym.items()}
    abs_qty = {k: sum(abs(x) for x in v) for k, v in by_sym.items()}
    return {"trade_fill_count_by_symbol": fill_count, "abs_filled_qty_by_symbol": abs_qty}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_file", type=Path)
    ap.add_argument("--key", default="trader", help="JSON key to store under in output")
    ap.add_argument("-o", "--out-json", type=Path, default=None, help="Append/update JSON file")
    args = ap.parse_args()

    data = json.loads(args.log_file.read_text(encoding="utf-8"))
    total, pnl_by_sym, n_act = parse_activities(data.get("activitiesLog", ""))
    th = data.get("tradeHistory") or []
    tr = parse_trades(th) if th else {}
    out: dict[str, Any] = {
        "log_file": str(args.log_file).replace("\\", "/"),
        "num_activity_rows": n_act,
        "num_trade_rows": len(th),
        "total_profit": total,
        "pnl_by_symbol": pnl_by_sym,
        **tr,
    }

    if args.out_json:
        payload: dict[str, Any] = {}
        if args.out_json.exists():
            try:
                payload = json.loads(args.out_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        payload[args.key] = out
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps({args.key: out}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
