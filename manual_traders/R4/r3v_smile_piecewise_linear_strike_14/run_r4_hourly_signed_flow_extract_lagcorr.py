#!/usr/bin/env python3
"""
Phase 2 — distributed lag (hour bucket): signed aggressive extract flow in hour H
(sum: +qty if buyer_agg else -qty if seller_agg on extract trades) vs extract **mid return**
over hour H+L (last mid in hour minus first mid in hour), L in {0,1,2,3}.

Pearson corr across pooled (day,hour) cells with valid return and n_trades>=5 in flow hour.
"""
from __future__ import annotations

import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
EXTRACT = "VELVETFRUIT_EXTRACT"
LAGS = (0, 1, 2, 3)


class Snap:
    __slots__ = ("ts", "mid", "bid", "ask")

    def __init__(self, ts: int, mid: float, bid: int, ask: int) -> None:
        self.ts = ts
        self.mid = mid
        self.bid = bid
        self.ask = ask


def load_extract_series(day: int) -> list[Snap]:
    rows: list[Snap] = []
    path = DATA / f"prices_round_4_day_{day}.csv"
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if int(r["day"]) != day or r["product"] != EXTRACT:
                continue
            ts = int(r["timestamp"])
            try:
                bb = int(float(r["bid_price_1"]))
                ba = int(float(r["ask_price_1"]))
            except (KeyError, ValueError):
                continue
            mid = float(r["mid_price"])
            rows.append(Snap(ts, mid, bb, ba))
    rows.sort(key=lambda s: s.ts)
    dedup: dict[int, Snap] = {}
    for s in rows:
        dedup[s.ts] = s
    return [dedup[t] for t in sorted(dedup)]


def snap_at(series: list[Snap], ts: int) -> Snap | None:
    tss = [s.ts for s in series]
    i = bisect.bisect_right(tss, ts) - 1
    if i < 0:
        return None
    return series[i]


def hour_of(ts: int) -> int:
    return (ts // 10000) % 24


def main() -> None:
    ext_series = {d: load_extract_series(d) for d in DAYS}
    # (day, hour) -> signed flow, n_trades
    flow: dict[tuple[int, int], list[float]] = defaultdict(list)
    ntr: dict[tuple[int, int], int] = defaultdict(int)

    for d in DAYS:
        ser = ext_series[d]
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["symbol"] != EXTRACT:
                    continue
                ts = int(r["timestamp"])
                h = hour_of(ts)
                try:
                    qty = float(r["quantity"])
                except (KeyError, ValueError):
                    continue
                sn = snap_at(ser, ts)
                if sn is None:
                    continue
                px = int(round(float(r["price"])))
                if px >= sn.ask:
                    flow[(d, h)].append(qty)
                elif px <= sn.bid:
                    flow[(d, h)].append(-qty)
                ntr[(d, h)] += 1

    flow_sum = {k: float(sum(v)) for k, v in flow.items()}

    # (day, hour) -> first and last mid in that hour from price tape
    def mids_in_hour(d: int, h: int) -> tuple[float | None, float | None]:
        ser = ext_series[d]
        first = last = None
        for s in ser:
            if hour_of(s.ts) != h:
                continue
            if first is None:
                first = s.mid
            last = s.mid
        return first, last

    returns: dict[tuple[int, int], float | None] = {}
    for d in DAYS:
        for h in range(24):
            a, b = mids_in_hour(d, h)
            if a is None or b is None:
                returns[(d, h)] = None
            else:
                returns[(d, h)] = float(b - a)

    out: dict = {"method": "hour=(ts//10000)%24; flow=sum signed qty on extract aggressive trades; return_h = last_mid-first_mid in hour", "lags": {}}
    for L in LAGS:
        xs: list[float] = []
        ys: list[float] = []
        for d in DAYS:
            for h in range(24):
                if ntr.get((d, h), 0) < 5:
                    continue
                hp = h + L
                if hp >= 24:
                    continue
                ret = returns.get((d, hp))
                if ret is None:
                    continue
                xs.append(flow_sum.get((d, h), 0.0))
                ys.append(ret)
        if len(xs) < 10:
            out["lags"][str(L)] = {"n": len(xs), "corr": None}
        else:
            c = float(np.corrcoef(np.array(xs), np.array(ys))[0, 1])
            out["lags"][str(L)] = {"n": len(xs), "corr": c}

    pth = OUT / "r4_hourly_signed_extract_flow_vs_return_lag.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
