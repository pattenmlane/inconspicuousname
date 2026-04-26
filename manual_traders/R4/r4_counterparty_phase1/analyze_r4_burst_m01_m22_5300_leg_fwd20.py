#!/usr/bin/env python3
"""At joint-tight timestamps: compare VEV_5300 K=20 forward mid when M01→Mark22
leg appears **on VEV_5300** vs burst where leg is only on other symbols.

Burst base: >=4 trade rows at (day, ts) AND some row has buyer Mark 01, seller Mark 22.
Stricter: same + at least one such row with symbol VEV_5300.

Round 4 days 1–3; K=20 on 5300 mid series (same construction as r4_gate_burst_5300_fwd_2x2).
"""
from __future__ import annotations

import csv
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_burst_m01_m22_5300_leg_fwd20.json"
DAYS = (1, 2, 3)
TH = 2.0
K = 20
BURST_MIN = 4
S5200, S5300 = "VEV_5200", "VEV_5300"


def load_trade_rows_by_ts():
    by: dict[tuple[int, int], list[tuple[str, str, str]]] = defaultdict(list)
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


def burst_base(rows: list[tuple[str, str, str]]) -> bool:
    if len(rows) < BURST_MIN:
        return False
    return any(b == "Mark 01" and s == "Mark 22" for _sym, b, s in rows)


def burst_5300_leg(rows: list[tuple[str, str, str]]) -> bool:
    if not burst_base(rows):
        return False
    return any(sym == S5300 and b == "Mark 01" and s == "Mark 22" for sym, b, s in rows)


def fwd(tss: list[int], mids: list[float], ts: int, k: int) -> float | None:
    i = bisect_right(tss, ts) - 1
    if i < 0:
        i = 0
    j = i + k
    if j >= len(mids):
        return None
    return mids[j] - mids[i]


def stat(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0, "mean": None, "frac_pos": None}
    n = len(xs)
    pos = sum(1 for x in xs if x > 0)
    return {"n": n, "mean": round(sum(xs) / n, 6), "frac_pos": round(pos / n, 6)}


def main() -> None:
    tr = load_trade_rows_by_ts()
    pooled_base: list[float] = []
    pooled_leg: list[float] = []
    pooled_basket_only: list[float] = []
    per_day: dict[str, dict[str, list[float]]] = {}

    for day in DAYS:
        per_day[str(day)] = {
            "tight_burst_any_m01_m22": [],
            "tight_burst_m01_m22_on_5300": [],
            "tight_burst_basket_only": [],
        }
        sp52: dict[int, float] = {}
        sp53: dict[int, float] = {}
        tss53: list[int] = []
        mids53: list[float] = []
        path = DATA / f"prices_round_4_day_{day}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day:
                    continue
                sym = row["product"]
                ts = int(row["timestamp"])
                bid = float(row["bid_price_1"])
                ask = float(row["ask_price_1"])
                sp = ask - bid if ask > bid else 0.0
                if sym == S5200:
                    sp52[ts] = sp
                elif sym == S5300:
                    sp53[ts] = sp
                    tss53.append(ts)
                    mids53.append(float(row["mid_price"]))
        if len(tss53) < K + 2:
            continue
        for ts in tss53:
            if ts not in sp52 or ts not in sp53:
                continue
            if sp52[ts] > TH or sp53[ts] > TH:
                continue
            rows = tr.get((day, ts), [])
            if not burst_base(rows):
                continue
            fk = fwd(tss53, mids53, ts, K)
            if fk is None:
                continue
            pool_key = "tight_burst_any_m01_m22"
            per_day[str(day)][pool_key].append(fk)
            pooled_base.append(fk)
            if burst_5300_leg(rows):
                per_day[str(day)]["tight_burst_m01_m22_on_5300"].append(fk)
                pooled_leg.append(fk)
            else:
                per_day[str(day)]["tight_burst_basket_only"].append(fk)
                pooled_basket_only.append(fk)

    out = {
        "K": K,
        "TH": TH,
        "BURST_MIN": BURST_MIN,
        "pooled": {
            "tight_burst_any_m01_m22": stat(pooled_base),
            "tight_burst_m01_m22_on_5300": stat(pooled_leg),
            "tight_burst_basket_only_m01_m22_elsewhere": stat(pooled_basket_only),
        },
        "per_day": {d: {k: stat(v) for k, v in cells.items()} for d, cells in per_day.items()},
        "interpretation": "If on-5300 leg subset has meaningfully higher mean_fwd20, conditioning entry on that leg (trader_v8) is supported by tape; if similar, stricter filter mainly reduces n.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
