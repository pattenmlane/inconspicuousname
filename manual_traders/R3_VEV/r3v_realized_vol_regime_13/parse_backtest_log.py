"""Parse prosperity4bt JSON log: total PnL from activitiesLog; tradeHistory aggregates."""
from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _total_pnl_activities(activities_log: str) -> float:
    r = csv.reader(io.StringIO(activities_log), delimiter=";")
    _ = next(r)
    best: dict[str, tuple[int, int, float]] = {}
    for row in r:
        if len(row) < 17:
            continue
        d, ts, pr = int(row[0]), int(row[1]), str(row[2])
        pl = float(row[-1])
        b = best.get(pr)
        if b is None or (d, ts) > (b[0], b[1]):
            best[pr] = (d, ts, pl)
    return float(sum(t[2] for t in best.values())) if best else 0.0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: parse_backtest_log.py <backtest.log>", file=sys.stderr)
        return 1
    p = Path(sys.argv[1])
    text = p.read_text(encoding="utf-8", errors="replace")
    o = json.loads(text.strip())
    th = o.get("tradeHistory") or []
    total_act = _total_pnl_activities(o.get("activitiesLog", ""))
    n_tr = len(th)
    sym_qty = Counter()
    sym_n = Counter()
    notional: dict[str, int] = defaultdict(int)
    for t in th:
        sym = str(t.get("symbol", ""))
        if not sym:
            continue
        q = int(t.get("quantity", 0) or 0)
        prc = int(t.get("price", 0) or 0)
        sym_n[sym] += 1
        sym_qty[sym] += abs(q)
        notional[sym] += abs(q) * abs(prc)
    top = sorted(notional.items(), key=lambda x: -x[1])[:10]
    top_s = [
        {
            "symbol": s,
            "trades": int(sym_n[s]),
            "qty": int(sym_qty[s]),
            "notional": float(notional[s]),
        }
        for s, _ in top
    ]
    out = {
        "total_profit_parsed": int(round(total_act)),
        "n_trades_total": n_tr,
        "gross_notional_total": float(sum(notional.values())),
        "top_symbols_by_notional": top_s,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
