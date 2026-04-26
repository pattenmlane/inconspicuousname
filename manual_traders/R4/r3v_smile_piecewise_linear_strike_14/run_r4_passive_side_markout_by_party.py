#!/usr/bin/env python3
"""
Phase 1 bullet 5 (population adverse-selection proxy): at each trade, infer **passive** side
as the counterparty that was **not** aggressive vs concurrent BBO on the **traded** symbol
(buyer_agg => passive seller; seller_agg => passive buyer; ambig excluded).

For each Mark name U, aggregate K=20 forward mid on the **traded** symbol (same as Phase-1
forward_mid) when U is passive. Report worst and best cells with n >= min_n.

This is **not** your MM inventory — it is "who is on the receiving side when the other
side lifts/crushes the touch."
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
    # (party, sym) -> list of fwd20 on traded sym when party was passive
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
                price = float(r["price"])
                px = int(round(price))
                buyer = (r.get("buyer") or "").strip()
                seller = (r.get("seller") or "").strip()
                if not buyer or not seller:
                    continue
                sn = snap_at(ser, ts)
                if sn is None:
                    continue
                if px >= sn.ask:
                    role = "buyer_agg"
                    passive = seller
                elif px <= sn.bid:
                    role = "seller_agg"
                    passive = buyer
                else:
                    continue
                fm = forward_mid(ser, ts, K)
                if fm is None:
                    continue
                delta = fm - sn.mid
                by_cell[(passive, sym)].append(delta)
                by_cell[(passive, "_ALL_")].append(delta)

    def sm(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
        }

    rows = []
    for (party, sym), xs in by_cell.items():
        if len(xs) < MIN_N:
            continue
        s = sm(xs)
        s["party"] = party
        s["symbol"] = sym
        rows.append(s)

    rows_by_mean = sorted(rows, key=lambda r: r["mean"])
    worst = rows_by_mean[:25]
    best = sorted(rows, key=lambda r: -r["mean"])[:25]

    out = {
        "method": "passive side = non-aggressor on traded sym BBO; ambig (inside spread) dropped; K=20 fwd on traded mid",
        "min_n": MIN_N,
        "n_cells_ge_min_n": len(rows),
        "worst_passive_mean_fwd20": worst,
        "best_passive_mean_fwd20": best,
    }
    pth = OUT / "r4_passive_side_markout_by_party_k20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
