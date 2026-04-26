#!/usr/bin/env python3
"""
Phase 1/2: VELVETFRUIT_EXTRACT trades only — participant (buyer or seller) × aggressor role
× burst (>=3 trades same day,ts) vs isolated. Extract forward mid K=20 at trade ts.

Summarizes pooled + per-day for selected high-signal cells from participant screen.
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
# (party, role) from Phase-1 screen emphasis
FOCAL = [
    ("Mark 67", "buyer_agg"),
    ("Mark 49", "buyer_agg"),
    ("Mark 22", "buyer_agg"),
    ("Mark 55", "buyer_agg"),
    ("Mark 14", "buyer_agg"),
]


class Snap:
    __slots__ = ("ts", "mid", "bid", "ask")

    def __init__(self, ts: int, mid: float, bid: int, ask: int) -> None:
        self.ts = ts
        self.mid = mid
        self.bid = bid
        self.ask = ask


def load_extract(day: int) -> list[Snap]:
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


def forward_mid(series: list[Snap], ts: int, k: int) -> float | None:
    tss = [s.ts for s in series]
    i = bisect.bisect_right(tss, ts)
    j = i + k - 1
    if j >= len(series):
        return None
    return float(series[j].mid)


def sm(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs)}


def main() -> None:
    burst_set: set[tuple[int, int]] = set()
    for d in DAYS:
        cnt: dict[int, int] = defaultdict(int)
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                cnt[int(r["timestamp"])] += 1
        for ts, c in cnt.items():
            if c >= 3:
                burst_set.add((d, ts))

    ext = {d: load_extract(d) for d in DAYS}
    # (party, role, burst01) -> list fwd20
    cells: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    by_day: dict[int, dict[tuple[str, str, int], list[float]]] = {
        d: defaultdict(list) for d in DAYS
    }

    for d in DAYS:
        ser = ext[d]
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["symbol"] != EXTRACT:
                    continue
                ts = int(r["timestamp"])
                sn = snap_at(ser, ts)
                if sn is None:
                    continue
                px = int(round(float(r["price"])))
                if px >= sn.ask:
                    role = "buyer_agg"
                elif px <= sn.bid:
                    role = "seller_agg"
                else:
                    continue
                fm = forward_mid(ser, ts, K)
                if fm is None:
                    continue
                delta = fm - sn.mid
                bur = 1 if (d, ts) in burst_set else 0
                buyer = (r.get("buyer") or "").strip()
                seller = (r.get("seller") or "").strip()
                # Same as Phase-1 screen: each side touched gets same (sym, aggressor role) bucket
                for party in (buyer, seller):
                    if not party:
                        continue
                    key = (party, role, bur)
                    cells[key].append(delta)
                    by_day[d][key].append(delta)

    out: dict = {"burst_def": ">=3 trades same (day,timestamp) any symbol", "K": K, "focal_cells": {}}
    for party, role in FOCAL:
        out["focal_cells"][f"{party}_{role}"] = {
            "pooled": {
                "burst": sm(cells.get((party, role, 1), [])),
                "isolated": sm(cells.get((party, role, 0), [])),
            },
            "per_day": {
                str(d): {
                    "burst": sm(by_day[d].get((party, role, 1), [])),
                    "isolated": sm(by_day[d].get((party, role, 0), [])),
                }
                for d in DAYS
            },
        }

    pth = OUT / "r4_participant_extract_burst_stratify_k20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
