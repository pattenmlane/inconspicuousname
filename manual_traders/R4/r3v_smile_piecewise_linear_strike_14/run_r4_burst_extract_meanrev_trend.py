#!/usr/bin/env python3
"""
After burst timestamp: extract mid fwd K in {5,20,100} vs **sign** of burst-window
extract mid move (last mid in burst minute minus first mid in burst minute — here
burst is single ts so use first snapshot at ts vs K=1 forward as micro proxy).

Simpler: burst_ts extract mid m0; fwd5, fwd20, fwd100; classify burst_direction =
sign(fwd5 - m0) vs sign(fwd20 - fwd5) for mean-revert (opposite) vs trend (same).

Actually implement:
- m0 = mid at burst ts (snap_at extract)
- r5 = fwd5 - m0, r20 = fwd20 - m0, r100 = fwd100 - m0
- trend_score = fraction of {r5,r20,r100} that is positive (if majority positive => up trend)
- meanrev = (r5>0 and r20<r5) or (r5<0 and r20>r5) as simple "5-tick overshoot then partial retrace"
  -> use: sign(r5) != sign(r20-r5)  (change in slope)

Report pooled counts for burst timestamps with valid fwd100.
"""
from __future__ import annotations

import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
EXTRACT = "VELVETFRUIT_EXTRACT"


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
    meanrev = 0
    trend = 0
    flat = 0
    n = 0
    for d, ts in sorted(burst_set):
        ser = ext[d]
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        m0 = sn.mid
        f5 = forward_mid(ser, ts, 5)
        f20 = forward_mid(ser, ts, 20)
        f100 = forward_mid(ser, ts, 100)
        if f100 is None or f5 is None or f20 is None:
            continue
        n += 1
        r5 = f5 - m0
        r20 = f20 - m0
        r100 = f100 - m0
        # mean-revert: initial move r5 and subsequent increment (r20-r5) opposite sign
        if r5 != 0 and (r20 - r5) * r5 < 0:
            meanrev += 1
        # trend: r5, r20-m0, r100-m0 same sign and |r100| > |r5|
        elif r5 != 0 and r20 * r5 > 0 and r100 * r5 > 0 and abs(r100) >= abs(r5):
            trend += 1
        else:
            flat += 1

    out = {
        "n_bursts_with_valid_fwd100": n,
        "mean_revert_pattern_count": meanrev,
        "trend_pattern_count": trend,
        "other_or_flat_count": flat,
        "frac_meanrev": meanrev / n if n else None,
        "frac_trend": trend / n if n else None,
        "definition": "meanrev: r5 and (r20-r5) opposite sign; trend: r5,r20,r100 same sign and |r100|>=|r5|",
    }
    pth = OUT / "r4_burst_extract_meanrev_trend_k.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
