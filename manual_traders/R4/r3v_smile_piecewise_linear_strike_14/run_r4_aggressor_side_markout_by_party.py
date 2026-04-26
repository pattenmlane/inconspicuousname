#!/usr/bin/env python3
"""
Complement to r4_passive_side_markout_by_party_k20.py: **aggressor** side only —
Mark U appears as buyer when buyer_agg or as seller when seller_agg; K=20 fwd on
**traded** symbol mid (Phase-1 forward_mid). ambig excluded.

Top/worst cells n>=min_n for Phase 1 bullet 5 (who lifts/crushes touch).
"""
from __future__ import annotations

import bisect
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
K = 20
MIN_N = 20


class Snap:
    __slots__ = ("ts", "mid", "bid", "ask", "spread")

    def __init__(self, ts: int, mid: float, bid: int, ask: int, spread: int) -> None:
        self.ts = ts
        self.mid = mid
        self.bid = bid
        self.ask = ask
        self.spread = spread


def load_prices(day: int) -> dict[str, list[Snap]]:
    by_sym: dict[str, list[Snap]] = defaultdict(list)
    path = DATA / f"prices_round_4_day_{day}.csv"
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if int(r["day"]) != day:
                continue
            sym = r["product"]
            ts = int(r["timestamp"])
            try:
                bb = int(float(r["bid_price_1"]))
                ba = int(float(r["ask_price_1"]))
            except (KeyError, ValueError):
                continue
            mid = float(r["mid_price"])
            by_sym[sym].append(Snap(ts, mid, bb, ba, ba - bb))
    for sym in by_sym:
        by_sym[sym].sort(key=lambda s: s.ts)
        dedup: dict[int, Snap] = {}
        for s in by_sym[sym]:
            dedup[s.ts] = s
        by_sym[sym] = [dedup[t] for t in sorted(dedup)]
    return dict(by_sym)


def snap_at(series: list[Snap], ts: int) -> Snap | None:
    tss = [s.ts for s in series]
    i = bisect.bisect_right(tss, ts) - 1
    if i < 0:
        return None
    return series[i]


def forward_mid(series: list[Snap], ts: int, k: int) -> float | None:
    tss = [s.ts for s in series]
    i = bisect.bisect_right(tss, ts)
    j = i + k - 1
    if j >= len(series):
        return None
    return float(series[j].mid)


def main() -> None:
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)

    for d in DAYS:
        by_sym = load_prices(d)
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                sym = r["symbol"]
                ser = by_sym.get(sym)
                if not ser:
                    continue
                ts = int(r["timestamp"])
                sn = snap_at(ser, ts)
                if sn is None:
                    continue
                px = int(round(float(r["price"])))
                if px >= sn.ask:
                    party = (r.get("buyer") or "").strip()
                    if not party:
                        continue
                elif px <= sn.bid:
                    party = (r.get("seller") or "").strip()
                    if not party:
                        continue
                else:
                    continue
                fm = forward_mid(ser, ts, K)
                if fm is None:
                    continue
                by_cell[(party, sym)].append(fm - sn.mid)

    def sm(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs)}

    rows = []
    for (party, sym), xs in by_cell.items():
        if len(xs) < MIN_N:
            continue
        s = sm(xs)
        s["party"] = party
        s["symbol"] = sym
        rows.append(s)

    worst = sorted(rows, key=lambda r: r["mean"])[:25]
    best = sorted(rows, key=lambda r: -r["mean"])[:25]

    out = {
        "method": "aggressor only: buyer if px>=ask else seller if px<=bid; K=20 fwd traded mid",
        "min_n": MIN_N,
        "worst_aggressor_mean_fwd20": worst,
        "best_aggressor_mean_fwd20": best,
    }
    pth = OUT / "r4_aggressor_side_markout_by_party_k20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
