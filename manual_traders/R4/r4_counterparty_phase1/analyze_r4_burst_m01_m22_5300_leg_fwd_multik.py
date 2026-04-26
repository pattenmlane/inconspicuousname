#!/usr/bin/env python3
"""Multi-horizon VEV_5300 fwd mid at joint-tight + M01→M22 burst, split by whether
the M01→Mark22 leg prints on VEV_5300 (Round 4 days 1–3).

K in {5,10,20,50} price rows; pooled + per-day means and frac_pos. Extends
r4_burst_m01_m22_5300_leg_fwd20.json with day-stability across horizons.
"""
from __future__ import annotations

import csv
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_burst_m01_m22_5300_leg_fwd_multik.json"
DAYS = (1, 2, 3)
TH = 2.0
KS = (5, 10, 20, 50)
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


def on_5300_leg(rows: list[tuple[str, str, str]]) -> bool:
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
    # (subset_name, k) -> list of fwd values for pooled
    pooled: dict[tuple[str, int], list[float]] = defaultdict(list)
    # day_str -> (subset_name, k) -> list
    per_day: dict[str, dict[tuple[str, int], list[float]]] = {
        str(d): defaultdict(list) for d in DAYS
    }

    for day in DAYS:
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
        max_k = max(KS)
        if len(tss53) < max_k + 2:
            continue
        ds = str(day)
        for ts in tss53:
            if ts not in sp52 or ts not in sp53:
                continue
            if sp52[ts] > TH or sp53[ts] > TH:
                continue
            rows = tr.get((day, ts), [])
            if not burst_base(rows):
                continue
            on_wing = on_5300_leg(rows)
            for k in KS:
                fk = fwd(tss53, mids53, ts, k)
                if fk is None:
                    continue
                key_any = ("any_m01_m22_burst", k)
                pooled[key_any].append(fk)
                per_day[ds][key_any].append(fk)
                if on_wing:
                    key_on = ("m01_m22_on_5300", k)
                    pooled[key_on].append(fk)
                    per_day[ds][key_on].append(fk)
                else:
                    key_b = ("m01_m22_basket_only", k)
                    pooled[key_b].append(fk)
                    per_day[ds][key_b].append(fk)

    def pack_cell(d: dict[tuple[str, int], list[float]]) -> dict:
        out = {}
        for name in ("any_m01_m22_burst", "m01_m22_on_5300", "m01_m22_basket_only"):
            out[name] = {f"K{k}": stat(d.get((name, k), [])) for k in KS}
        return out

    out_json = {
        "TH": TH,
        "BURST_MIN": BURST_MIN,
        "KS": list(KS),
        "pooled": pack_cell(pooled),
        "per_day": {d: pack_cell(per_day[d]) for d in ("1", "2", "3")},
        "note": "any_m01_m22_burst equals union of on_5300 and basket_only at same ts (disjoint split).",
    }
    OUT.write_text(json.dumps(out_json, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
