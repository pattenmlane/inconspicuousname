#!/usr/bin/env python3
"""Copy of R3 summarizer: JSON log -> end PnL + submission trade stats."""
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
    reader = csv.DictReader(io.StringIO(alog), delimiter=";")
    for row in reader:
        prod = row.get("product", "")
        if not prod or "profit_and_loss" not in row:
            continue
        try:
            last_pnl[prod] = float(row["profit_and_loss"])
        except (ValueError, KeyError):
            pass
    # Round 4 tape rows always have buyer+seller (Mark IDs). Count **our** fills only.
    th = [
        t
        for t in data.get("tradeHistory", [])
        if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"
    ]
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
        "n_submission_trade_rows": len(th),
        "submission_qty_by_symbol": dict(qty_by),
        "submission_trades_by_symbol": dict(ntr),
    }


def main() -> None:
    p = Path(sys.argv[1])
    print(json.dumps(summarize(p), indent=2))


if __name__ == "__main__":
    main()
