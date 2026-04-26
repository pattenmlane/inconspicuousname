#!/usr/bin/env python3
"""Summarize own fills from prosperity4bt JSON: tradeHistory rows where we are SUBMISSION (buy or sell)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    args = p.parse_args()
    data = json.loads(args.json_path.read_text())
    th = data.get("tradeHistory") or []
    own_buys = [t for t in th if isinstance(t, dict) and t.get("buyer") == "SUBMISSION"]
    own_sells = [t for t in th if isinstance(t, dict) and t.get("seller") == "SUBMISSION"]
    own = own_buys + own_sells
    qty_by_sym: Counter[str] = Counter()
    for t in own:
        sym = t.get("symbol")
        q = int(t.get("quantity") or 0)
        if sym is not None and q:
            qty_by_sym[str(sym)] += abs(q)
    out = {
        "source": args.json_path.name,
        "num_submission_trades": len(own),
        "num_buy_submission": len(own_buys),
        "num_sell_submission": len(own_sells),
        "filled_abs_qty_by_symbol": dict(sorted(qty_by_sym.items(), key=lambda x: -x[1])),
    }
    text = json.dumps(out, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
