#!/usr/bin/env python3
"""
Tape: trades where buyer=='Mark 01' and seller=='Mark 22' on any VEV_* strike.
Extract forward mid K in {5,20,100} from trade timestamp; stratify by joint-tight
(5200+5300 L1<=2) at trade ts.

Phase-1 snap_at / forward_mid conventions.
"""
from __future__ import annotations

import bisect
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
K_LIST = (5, 20, 100)
EXTRACT = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
TH = 2
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
    from collections import defaultdict

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
    return {"n": len(xs), "mean": statistics.mean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0}


def main() -> None:
    by_day = {d: load_prices(d) for d in DAYS}
    ext = {d: by_day[d][EXTRACT] for d in DAYS}

    # (tight_flag, K) -> list of extract fwd deltas
    from collections import defaultdict

    pooled: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_sym: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_trades = 0

    for d in DAYS:
        ser_e = ext[d]
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                sym = r["symbol"]
                if not sym.startswith("VEV_"):
                    continue
                if (r.get("buyer") or "").strip() != BUYER:
                    continue
                if (r.get("seller") or "").strip() != SELLER:
                    continue
                ts = int(r["timestamp"])
                sn = snap_at(ser_e, ts)
                if sn is None:
                    continue
                jt = joint_tight(by_day[d], ts)
                if jt is None:
                    continue
                tight = 1 if jt else 0
                n_trades += 1
                for K in K_LIST:
                    fm = forward_mid(ser_e, ts, K)
                    if fm is None:
                        continue
                    delta = fm - sn.mid
                    pooled[(tight, K)].append(delta)
                    by_sym[sym][(tight, K)].append(delta)

    out = {
        "filter": "buyer Mark 01, seller Mark 22, symbol VEV_*",
        "n_prints_used": n_trades,
        "pooled_by_tight_and_K": {
            f"tight_{ti}_K{kk}": summarize(vv)
            for (ti, kk), vv in sorted(pooled.items())
        },
        "by_vev_symbol_top_cells": {
            sym: {
                f"tight_{ti}_K{kk}": summarize(vv)
                for (ti, kk), vv in sorted(d.items())
                if len(vv) >= 3
            }
            for sym, d in sorted(by_sym.items())
            if any(len(v) >= 3 for v in d.values())
        },
    }
    pth = OUT / "r4_m01_m22_vev_extract_markout_gate.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
