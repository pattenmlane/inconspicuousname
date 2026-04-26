#!/usr/bin/env python3
"""Summarize prosperity4bt JSON log: total PnL from stdout-style parse of activitiesLog tail, fill stats from tradeHistory."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def total_profit_from_activities(activities_log: str) -> tuple[int | None, dict[int, float], dict[int, float]]:
    """Merged multi-day logs from `--out`: `profit_and_loss` is **cumulative** across csv days.

    Let S(d) = sum of profit_and_loss (all products) at the max timestamp for csv day column d.
    Incremental PnL on day d is S(d) - S(d-1); **total** PnL over the run is S(D) for the last csv day D
    (matches backtester stdout).
    """
    lines = activities_log.strip().split("\n")
    if len(lines) < 2:
        return None, {}, {}
    header = lines[0].split(";")
    if "profit_and_loss" not in header:
        return None, {}, {}
    pi = header.index("profit_and_loss")
    by_day_ts: dict[int, dict[int, list[list[str]]]] = {}
    for line in lines[1:]:
        parts = line.split(";")
        if len(parts) <= pi:
            continue
        try:
            day = int(parts[0])
            ts = int(parts[1])
        except ValueError:
            continue
        by_day_ts.setdefault(day, {}).setdefault(ts, []).append(parts)
    cumulative_end: dict[int, float] = {}
    for day, tsmap in sorted(by_day_ts.items()):
        last_ts = max(tsmap)
        rows = tsmap[last_ts]
        cumulative_end[day] = sum(float(r[pi]) for r in rows)
    days_sorted = sorted(cumulative_end)
    incremental: dict[int, float] = {}
    prev = 0.0
    for d in days_sorted:
        cur = cumulative_end[d]
        incremental[d] = cur - prev
        prev = cur
    total = int(round(cumulative_end[days_sorted[-1]])) if days_sorted else None
    return total, incremental, cumulative_end


def submission_fills_by_symbol(trade_history: list[dict], sym: str | None) -> dict:
    by_sym: dict[str, dict[str, float]] = defaultdict(
        lambda: {"fills": 0, "abs_qty": 0.0, "net_qty": 0.0}
    )
    n_sub = 0
    for t in trade_history:
        symbol = t.get("symbol", "")
        if sym and symbol != sym:
            continue
        buyer = t.get("buyer", "")
        seller = t.get("seller", "")
        if buyer != "SUBMISSION" and seller != "SUBMISSION":
            continue
        n_sub += 1
        q = float(t.get("quantity", 0))
        cell = by_sym[symbol]
        cell["fills"] += 1
        cell["abs_qty"] += abs(q)
        if buyer == "SUBMISSION":
            cell["net_qty"] += q
        else:
            cell["net_qty"] -= q
    out_by = {k: {"fills": int(v["fills"]), "abs_qty": v["abs_qty"], "net_qty": v["net_qty"]} for k, v in by_sym.items()}
    return {"n_submission_trade_events": n_sub, "by_symbol": out_by}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_json", type=Path, help="Backtest --out JSON log")
    ap.add_argument("--trader", required=True, help="e.g. trader_v9.py")
    ap.add_argument("--match-trades", required=True, choices=("worse", "all", "none"))
    ap.add_argument("--product-symbol", default="VELVETFRUIT_EXTRACT", help="Filter fill summary to this symbol only")
    args = ap.parse_args()
    data = json.loads(args.log_json.read_text())
    act = data.get("activitiesLog") or ""
    tp, inc_by_day, cum_end = total_profit_from_activities(act)
    fills = submission_fills_by_symbol(data.get("tradeHistory") or [], args.product_symbol)
    summary = {
        "trader": args.trader,
        "match_trades": args.match_trades,
        "total_profit_merged_activities": tp,
        "profit_incremental_by_csv_day": {str(k): v for k, v in sorted(inc_by_day.items())},
        "profit_cumulative_end_by_csv_day": {str(k): v for k, v in sorted(cum_end.items())},
        **fills,
    }
    out_path = args.log_json.with_name(
        args.log_json.name.replace("_backtest.log", "_fill_summary.json").replace(".log", "_fill_summary.json")
    )
    if out_path == args.log_json:
        out_path = args.log_json.with_suffix(".fill_summary.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if tp is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
