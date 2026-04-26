#!/usr/bin/env python3
"""
Same-timestamp orchestration: (day, ts) with Mark01->Mark22 on any VEV_* vs presence of
VELVETFRUIT_EXTRACT trade; extract forward mid K=20 from that ts (Phase-1 forward_mid).

Also: among M01->M22 VEV timestamps, share that also have extract print; extract fwd20 mean.
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
BUYER = "Mark 01"
SELLER = "Mark 22"


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
    by_day = {d: load_prices(d) for d in DAYS}
    ext = {d: by_day[d][EXTRACT] for d in DAYS}

    # (day, ts) -> set of symbols traded
    syms_at: dict[tuple[int, int], set[str]] = defaultdict(set)
    m01_m22_vev_ts: set[tuple[int, int]] = set()
    extract_ts: set[tuple[int, int]] = set()

    for d in DAYS:
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                ts = int(r["timestamp"])
                sym = r["symbol"]
                key = (d, ts)
                syms_at[key].add(sym)
                if sym == EXTRACT:
                    extract_ts.add(key)
                if sym.startswith("VEV_"):
                    if (r.get("buyer") or "").strip() == BUYER and (r.get("seller") or "").strip() == SELLER:
                        m01_m22_vev_ts.add(key)

    both = m01_m22_vev_ts & extract_ts
    vev_only = m01_m22_vev_ts - extract_ts
    n_vev_ts = len(m01_m22_vev_ts)
    n_both = len(both)
    n_vev_only = len(vev_only)

    fwd_both: list[float] = []
    fwd_vev_only: list[float] = []

    for d, ts in both:
        ser = ext[d]
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        fm = forward_mid(ser, ts, K)
        if fm is None:
            continue
        fwd_both.append(fm - sn.mid)

    for d, ts in vev_only:
        ser = ext[d]
        sn = snap_at(ser, ts)
        if sn is None:
            continue
        fm = forward_mid(ser, ts, K)
        if fm is None:
            continue
        fwd_vev_only.append(fm - sn.mid)

    def sm(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean": statistics.mean(xs)}

    out = {
        "n_timestamps_m01_m22_vev": n_vev_ts,
        "n_timestamps_also_extract_trade": n_both,
        "n_timestamps_vev_only_no_extract_print": n_vev_only,
        "frac_vev_ts_with_extract_print": n_both / n_vev_ts if n_vev_ts else None,
        "extract_fwd20_at_vev_ts": {
            "when_extract_also_prints_same_ts": sm(fwd_both),
            "when_no_extract_print_same_ts": sm(fwd_vev_only),
        },
    }
    pth = OUT / "r4_orchestration_m01_m22_vev_extract_same_ts.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
