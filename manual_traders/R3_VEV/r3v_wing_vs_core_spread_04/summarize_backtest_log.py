#!/usr/bin/env python3
"""Parse prosperity4bt JSON log: end-of-tape PnL by product + submission trade stats."""
from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path


def summarize(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    alog = data.get("activitiesLog", "")
    last_pnl: dict[str, float] = {}
    n_rows = 0
    reader = csv.DictReader(io.StringIO(alog), delimiter=";")
    for row in reader:
        n_rows += 1
        prod = row.get("product", "")
        if not prod or "profit_and_loss" not in row:
            continue
        try:
            last_pnl[prod] = float(row["profit_and_loss"])
        except (ValueError, KeyError):
            pass
    th = [t for t in data.get("tradeHistory", []) if t.get("buyer") or t.get("seller")]
    qty_by: Counter[str] = Counter()
    ntr: Counter[str] = Counter()
    for t in th:
        s = t.get("symbol", "")
        qty_by[s] += int(t.get("quantity", 0))
        ntr[s] += 1
    total = sum(last_pnl.values())
    return {
        "log_file": str(path),
        "total_profit_end": total,
        "products_pnl_end": {k: last_pnl[k] for k in sorted(last_pnl.keys())},
        "n_trade_rows_total": n_rows,
        "n_submission_trade_rows": len(th),
        "submission_qty_by_symbol": dict(qty_by),
        "submission_trades_by_symbol": dict(ntr),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: summarize_backtest_log.py <backtest.log> [out.json]", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    out = summarize(path)
    if len(sys.argv) >= 3:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
