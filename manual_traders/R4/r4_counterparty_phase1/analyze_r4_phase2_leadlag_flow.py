#!/usr/bin/env python3
"""Phase 2.3: Cross-instrument signed flow vs extract mid changes (hour buckets).

Per (day, hour): sum signed qty on trades (buyer-initiated +qty at ask, seller-initiated -qty at bid heuristic),
aggregate VEV_5300 vs VELVETFRUIT_EXTRACT; correlate hour returns of extract mid with lagged 5300 flow.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_phase2_leadlag_hourly.json"
DAYS = (1, 2, 3)
SYMS = ("VEV_5300", "VELVETFRUIT_EXTRACT")


def load_mid_series(day: int, sym: str) -> dict[int, float]:
    mp: dict[int, float] = {}
    path = DATA / f"prices_round_4_day_{day}.csv"
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            if int(row["day"]) != day or row["product"] != sym:
                continue
            mp[int(row["timestamp"])] = float(row["mid_price"])
    return mp


def trade_sign(row: dict, bid: float, ask: float) -> int:
    p = float(row["price"])
    q = int(float(row["quantity"]))
    if p >= ask - 1e-9:
        return q
    if p <= bid + 1e-9:
        return -q
    return 0


def main() -> None:
    hourly: dict[tuple, dict[str, float]] = defaultdict(
        lambda: {"f5300": 0.0, "fex": 0.0, "r5300": 0.0, "rex": 0.0, "n": 0}
    )
    for day in DAYS:
        mids = {s: load_mid_series(day, s) for s in SYMS}
        # join trades to nearest price at same ts for book
        books: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
        path = DATA / f"prices_round_4_day_{day}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day:
                    continue
                sym = row["product"]
                if sym not in SYMS:
                    continue
                ts = int(row["timestamp"])
                bid = float(row["bid_price_1"])
                ask = float(row["ask_price_1"])
                books[ts][sym] = (bid, ask)
        tp = DATA / f"trades_round_4_day_{day}.csv"
        with open(tp, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                sym = row["symbol"]
                if sym not in SYMS:
                    continue
                ts = int(row["timestamp"])
                if ts not in books or sym not in books[ts]:
                    continue
                bid, ask = books[ts][sym]
                h = ts // 100000
                key = (day, h)
                sg = trade_sign(row, bid, ask)
                if sym == "VEV_5300":
                    hourly[key]["f5300"] += sg
                else:
                    hourly[key]["fex"] += sg
        # hour mid returns: first vs last mid in hour for extract
        for h in range(0, 11):
            key = (day, h)
            ts_list = sorted(
                t for t in mids["VELVETFRUIT_EXTRACT"] if t // 100000 == h
            )
            if len(ts_list) < 2:
                continue
            m0 = mids["VELVETFRUIT_EXTRACT"][ts_list[0]]
            m1 = mids["VELVETFRUIT_EXTRACT"][ts_list[-1]]
            hourly[key]["rex"] = m1 - m0
            ts53 = sorted(t for t in mids["VEV_5300"] if t // 100000 == h)
            if len(ts53) >= 2:
                hourly[key]["r5300"] = (
                    mids["VEV_5300"][ts53[-1]] - mids["VEV_5300"][ts53[0]]
                )

    # correlation f5300 (hour h) vs rex (hour h+1) pooled
    xs, ys = [], []
    for day in DAYS:
        for h in range(0, 9):
            k0 = (day, h)
            k1 = (day, h + 1)
            if k0 not in hourly or k1 not in hourly:
                continue
            xs.append(hourly[k0]["f5300"])
            ys.append(hourly[k1]["rex"])
    if len(xs) >= 5:
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        deny = math.sqrt(sum((y - my) ** 2 for y in ys))
        corr = num / (denx * deny) if denx > 0 and deny > 0 else None
    else:
        corr = None

    out = {
        "n_hour_pairs_lag1": len(xs),
        "corr_f5300_h_vs_extract_return_h_plus_1": round(corr, 4) if corr is not None else None,
        "note": "Coarse hour buckets (ts//100000); sign from touch heuristic.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
