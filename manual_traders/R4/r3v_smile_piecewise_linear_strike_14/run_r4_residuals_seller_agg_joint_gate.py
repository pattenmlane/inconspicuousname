#!/usr/bin/env python3
"""
Extract trades, seller_agg only (trade price <= concurrent extract bid): residual =
fwd20 - mean(buyer,seller,EXTRACT) in seller_agg-only baseline cells (n>=5).

Stratify by joint-tight at trade ts — mirror of run_r4_residuals_buyer_agg_joint_gate.py.
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
EXTRACT = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
TH = 2


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


def joint_tight(day_sym: dict[str, list[Snap]], ts: int) -> bool | None:
    a = snap_at(day_sym.get(S5200, []), ts)
    b = snap_at(day_sym.get(S5300, []), ts)
    if a is None or b is None:
        return None
    return a.spread <= TH and b.spread <= TH


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    m = statistics.mean(xs)
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return {"n": len(xs), "mean": m, "std": sd}


def main() -> None:
    by_day = {d: load_prices(d) for d in DAYS}
    base_cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for d in DAYS:
        ser_e = by_day[d].get(EXTRACT, [])
        if not ser_e:
            continue
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["symbol"] != EXTRACT:
                    continue
                ts = int(r["timestamp"])
                price = float(r["price"])
                px = int(round(price))
                buyer = (r.get("buyer") or "").strip()
                seller = (r.get("seller") or "").strip()
                if not buyer or not seller:
                    continue
                sn = snap_at(ser_e, ts)
                if sn is None or px > sn.bid:
                    continue
                fm = forward_mid(ser_e, ts, K)
                if fm is None:
                    continue
                key = (buyer, seller, EXTRACT)
                base_cells[key].append(fm - sn.mid)

    baseline = {k: statistics.mean(v) for k, v in base_cells.items() if len(v) >= 5}

    tight_res: list[float] = []
    loose_res: list[float] = []
    per_day: dict[int, dict[str, list[float]]] = {
        d: {"tight": [], "loose": []} for d in DAYS
    }

    for d in DAYS:
        ser_e = by_day[d].get(EXTRACT, [])
        if not ser_e:
            continue
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["symbol"] != EXTRACT:
                    continue
                ts = int(r["timestamp"])
                price = float(r["price"])
                px = int(round(price))
                buyer = (r.get("buyer") or "").strip()
                seller = (r.get("seller") or "").strip()
                if not buyer or not seller:
                    continue
                sn = snap_at(ser_e, ts)
                if sn is None or px > sn.bid:
                    continue
                key = (buyer, seller, EXTRACT)
                if key not in baseline:
                    continue
                fm = forward_mid(ser_e, ts, K)
                if fm is None:
                    continue
                delta = fm - sn.mid
                res = delta - baseline[key]
                jt = joint_tight(by_day[d], ts)
                if jt is None:
                    continue
                if jt:
                    tight_res.append(res)
                    per_day[d]["tight"].append(res)
                else:
                    loose_res.append(res)
                    per_day[d]["loose"].append(res)

    out = {
        "method": "seller_agg extract only (px<=bid); baseline = mean fwd20 per (buyer,seller,EXTRACT) if n>=5 in seller_agg cell",
        "n_baseline_cells": len(baseline),
        "pooled_residuals": {
            "joint_tight": summarize(tight_res),
            "joint_loose": summarize(loose_res),
        },
        "per_day": {str(d): {k: summarize(v) for k, v in per_day[d].items()} for d in DAYS},
    }
    pth = OUT / "r4_residuals_seller_agg_joint_gate.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
