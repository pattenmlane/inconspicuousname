#!/usr/bin/env python3
"""Round 4: burst **orchestrator** proxy vs forward mids (Phase 1 burst structure + Phase 3 gate).

For each (day, timestamp) with >=4 trade rows:
- joint_tight: VEV_5200 and VEV_5300 L1 spread <= TH at that ts (from prices file)
- dominant **buyer** across symbols (count rows per buyer; tie -> lexicographically first)
- whether Mark 01 is buyer on at least one row (basket leg)
- K=20 fwd on VEV_5300 mid and VELVETFRUIT_EXTRACT mid (row index alignment per product)

Outputs JSON: mean fwd by (tight?, dominant_buyer top bucket), and Mark01-present vs not.
"""
from __future__ import annotations

import csv
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_burst_orchestrator_fwd.json"
DAYS = (1, 2, 3)
TH = 2.0
BURST_MIN = 4
K = 20
S5200, S5300 = "VEV_5200", "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"


def load_trades_by_ts():
    by: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        with open(p, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                sym = str(row["symbol"])
                b = str(row["buyer"]).strip()
                s = str(row["seller"]).strip()
                by[(d, int(row["timestamp"]))].append((sym, b, s))
    return by


def load_joint_tight_flag():
    """(d, ts) -> True if s5200<=TH and s5300<=TH"""
    sp52: dict[tuple[int, int], float] = {}
    sp53: dict[tuple[int, int], float] = {}
    for d in DAYS:
        path = DATA / f"prices_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != d:
                    continue
                ts = int(row["timestamp"])
                sym = row["product"]
                bid = float(row["bid_price_1"])
                ask = float(row["ask_price_1"])
                sp = ask - bid if ask > bid else 0.0
                if sym == S5200:
                    sp52[(d, ts)] = sp
                elif sym == S5300:
                    sp53[(d, ts)] = sp
    out = {}
    keys = set(sp52) & set(sp53)
    for k in keys:
        out[k] = sp52[k] <= TH and sp53[k] <= TH
    return out


def fwd_series(tss: list[int], mids: list[float], ts: int) -> float | None:
    i = bisect_right(tss, ts) - 1
    if i < 0:
        i = 0
    j = i + K
    if j >= len(mids):
        return None
    return mids[j] - mids[i]


def stat(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None}
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 6)}


def main() -> None:
    tr = load_trades_by_ts()
    tight = load_joint_tight_flag()
    # collect fwd per bucket per day
    buckets: dict[str, list[float]] = defaultdict(list)
    buck53: dict[str, list[float]] = defaultdict(list)

    for day in DAYS:
        tss53, mids53 = [], []
        tss_ex, mids_ex = [], []
        path = DATA / f"prices_round_4_day_{day}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day:
                    continue
                sym = row["product"]
                ts = int(row["timestamp"])
                mp = float(row["mid_price"])
                if sym == S5300:
                    tss53.append(ts)
                    mids53.append(mp)
                elif sym == EX:
                    tss_ex.append(ts)
                    mids_ex.append(mp)
        if len(tss53) < K + 2:
            continue
        for ts in sorted(set(tss53) | set(t for (d, t) in tr if d == day)):
            if ts not in tss53:
                continue
            rows = tr.get((day, ts), [])
            if len(rows) < BURST_MIN:
                continue
            fk53 = fwd_series(tss53, mids53, ts)
            fkex = fwd_series(tss_ex, mids_ex, ts) if len(tss_ex) >= K + 2 else None
            if fk53 is None:
                continue
            buyers = [b for _sym, b, _s in rows]
            dom = Counter(buyers).most_common(1)[0][0]
            m01_buy = any(b == "Mark 01" for b in buyers)
            jt = tight.get((day, ts), False)
            key_a = f"tight={jt}_dom={dom}"
            key_b = f"tight={jt}_m01_buy_row={'y' if m01_buy else 'n'}"
            for lab in (key_a, key_b):
                buck53[lab].append(fk53)
                if fkex is not None:
                    buckets[lab].append(fkex)

    # aggregate Mark 01 dom vs other when tight
    mark01_tight_53: list[float] = []
    other_tight_53: list[float] = []
    per_day_dom: dict[str, dict[str, list[float]]] = {
        str(d): {"mark01_dom": [], "other_dom": []} for d in DAYS
    }
    for day in DAYS:
        tss53, mids53 = [], []
        path = DATA / f"prices_round_4_day_{day}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day or row["product"] != S5300:
                    continue
                tss53.append(int(row["timestamp"]))
                mids53.append(float(row["mid_price"]))
        for ts in tss53:
            rows = tr.get((day, ts), [])
            if len(rows) < BURST_MIN or not tight.get((day, ts), False):
                continue
            buyers = [b for _sym, b, _s in rows]
            dom = Counter(buyers).most_common(1)[0][0]
            fk = fwd_series(tss53, mids53, ts)
            if fk is None:
                continue
            if dom == "Mark 01":
                mark01_tight_53.append(fk)
                per_day_dom[str(day)]["mark01_dom"].append(fk)
            else:
                other_tight_53.append(fk)
                per_day_dom[str(day)]["other_dom"].append(fk)

    out = {
        "TH": TH,
        "BURST_MIN": BURST_MIN,
        "K": K,
        "definition": "dominant buyer = mode of buyer field over rows at (day,ts); burst = len(rows)>=BURST_MIN",
        "joint_tight_gated": {
            "fwd20_5300_mark01_dom_tight": stat(mark01_tight_53),
            "fwd20_5300_other_dom_tight": stat(other_tight_53),
            "per_day": {
                d: {
                    "mark01_dom": stat(v["mark01_dom"]),
                    "other_dom": stat(v["other_dom"]),
                }
                for d, v in per_day_dom.items()
            },
        },
        "cells_ex_fwd20_sample": {k: stat(v) for k, v in sorted(buckets.items()) if len(v) >= 15},
        "cells_5300_fwd20_sample": {k: stat(v) for k, v in sorted(buck53.items()) if len(v) >= 15},
        "note": "Full cell dump truncated to n>=15; see joint_tight_gated for Mark01-as-mode-buyer vs rest under tight+burst timestamps on 5300 clock.",
    }
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
