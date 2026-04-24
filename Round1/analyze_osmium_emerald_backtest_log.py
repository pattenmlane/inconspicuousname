#!/usr/bin/env python3
"""
Parse a prosperity4bt JSON log (written with --out) and attribute our
ASH_COATED_OSMIUM submission trades to logged intents.

Buckets (PnL = signed cash from fills: buys -price*qty, sells +price*qty):
  - ask_taking: take_buy_best_ask_below_fv
  - bid_taking: take_sell_best_bid_above_fv
  - market_making: passive_mm_buy | passive_mm_sell
  - clear_at_fv: clear_long_sell_at_ceil_fv | clear_short_buy_at_floor_fv
  - unmatched: could not align trade to an intent at that timestamp

Usage:
  python3 Round1/analyze_osmium_emerald_backtest_log.py backtests/foo.log [more.log ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

LOG_PREFIX = "OSMIUM_EM_JSON"
SYMBOL = "ASH_COATED_OSMIUM"


def _kind_bucket(kind: str) -> str:
    if kind == "take_buy_best_ask_below_fv":
        return "ask_taking"
    if kind == "take_sell_best_bid_above_fv":
        return "bid_taking"
    if kind in ("passive_mm_buy", "passive_mm_sell"):
        return "market_making"
    if kind in ("clear_long_sell_at_ceil_fv", "clear_short_buy_at_floor_fv"):
        return "clear_at_fv"
    return "other"


def _parse_intents_from_lambda(lambda_log: str) -> list[dict]:
    out: list[dict] = []
    if not lambda_log:
        return out
    for line in lambda_log.splitlines():
        if not line.startswith(LOG_PREFIX):
            continue
        raw = line[len(LOG_PREFIX) :]
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("event") != "intent":
            continue
        out.append(obj)
    return out


def _trade_cashflow(tr: dict) -> float | None:
    if tr.get("symbol") != "ASH_COATED_OSMIUM":
        return None
    p, q = int(tr["price"]), int(tr["quantity"])
    if tr.get("buyer") == "SUBMISSION":
        return -p * q
    if tr.get("seller") == "SUBMISSION":
        return p * q
    return None


def _final_pnl_from_activities(activities_log: str) -> float | None:
    lines = (activities_log or "").strip().split("\n")
    if len(lines) < 2:
        return None
    last: float | None = None
    for line in lines[1:]:
        parts = line.split(";")
        if len(parts) > 3 and parts[2] == SYMBOL:
            try:
                last = float(parts[-1])
            except ValueError:
                pass
    return last


def analyze_log(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    final_pnl = _final_pnl_from_activities(data.get("activitiesLog", ""))

    # intents per timestamp (FIFO)
    intents_by_ts: dict[int, list[dict]] = defaultdict(list)
    for row in data.get("logs", []):
        ts = int(row["timestamp"])
        for it in _parse_intents_from_lambda(row.get("lambdaLog") or ""):
            intents_by_ts[ts].append(it)

    for ts in intents_by_ts:
        intents_by_ts[ts] = list(intents_by_ts[ts])

    pnl_by = defaultdict(float)
    qty_by = defaultdict(int)
    unmatched_cash = 0.0
    unmatched_qty = 0

    trades = [
        t
        for t in data.get("tradeHistory", [])
        if t.get("symbol") == SYMBOL
        and (t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION")
    ]
    trades.sort(key=lambda t: (int(t["timestamp"]), t.get("buyer") or "", t.get("seller") or ""))

    for tr in trades:
        cash = _trade_cashflow(tr)
        if cash is None:
            continue
        ts = int(tr["timestamp"])
        price = int(tr["price"])
        qty = int(tr["quantity"])
        q_need = qty

        queue = intents_by_ts.get(ts, [])
        placed = False
        i = 0
        while i < len(queue) and q_need > 0:
            it = queue[i]
            ip = int(it["price"])
            iq = int(it["qty"])
            if ip != price or iq <= 0:
                i += 1
                continue
            take = min(q_need, iq)
            frac = take / qty if qty else 1.0
            bucket = _kind_bucket(str(it.get("kind", "")))
            pnl_by[bucket] += cash * (take / qty) if qty else cash
            qty_by[bucket] += take
            it["qty"] = iq - take
            q_need -= take
            placed = True
            if it["qty"] <= 0:
                queue.pop(i)
            else:
                i += 1

        if not placed or q_need > 0:
            rem_frac = (q_need / qty) if qty else 1.0
            unmatched_cash += cash * rem_frac if qty else cash
            unmatched_qty += q_need

    realized_cash = sum(pnl_by.values())
    return {
        "path": str(path),
        "submission_trades_osmium": len(trades),
        "final_reported_pnl_osmium (activity log)": final_pnl,
        "sum_realized_cash_from_fills (by bucket)": realized_cash,
        "pnl_by_bucket (signed cash at each fill)": dict(pnl_by),
        "qty_by_bucket": dict(qty_by),
        "unmatched_cash": unmatched_cash,
        "unmatched_qty": unmatched_qty,
        "note": "Bucket sums are fill cashflows only; backtest PnL includes MTM on open inventory at last tick.",
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("Usage: python3 analyze_osmium_emerald_backtest_log.py <backtest.log> [...]", file=sys.stderr)
        sys.exit(1)
    for p in paths:
        if not p.is_file():
            print(f"Skip missing: {p}", file=sys.stderr)
            continue
        r = analyze_log(p)
        print(f"\n=== {p.name} ===")
        print(f"  path: {r['path']}")
        for k, v in r.items():
            if k in ("path",):
                continue
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in sorted(v.items()):
                    print(f"    {kk}: {vv:,.2f}" if isinstance(vv, float) else f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
