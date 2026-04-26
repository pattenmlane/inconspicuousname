#!/usr/bin/env python3
"""VEV_5300 L1 spread by regime: joint tight gate × M01→M22 burst (Round 4 days 1–3).

Motivation for trader_v7: passive exit at max(ask−1, bid) when spread≥2 needs
spread≥2 to be frequent at regime boundaries; compare P(spread≥2) and mean spread
across (tight∧burst), (tight∧¬burst), (¬tight).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_5300_spread_by_gate_burst.json"
DAYS = (1, 2, 3)
TH = 2.0
BURST_MIN = 4
S5200, S5300 = "VEV_5200", "VEV_5300"


def load_burst_flag():
    by: dict[tuple[int, int], list] = defaultdict(list)
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        with open(p, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                by[(d, int(row["timestamp"]))].append(row)
    out: dict[tuple[int, int], bool] = {}
    for key, rows in by.items():
        if len(rows) < BURST_MIN:
            out[key] = False
            continue
        out[key] = any(
            str(x["buyer"]).strip() == "Mark 01" and str(x["seller"]).strip() == "Mark 22"
            for x in rows
        )
    return out


def summarize(spreads: list[float]) -> dict:
    if not spreads:
        return {"n": 0, "mean_spread": None, "frac_spread_ge_2": None}
    n = len(spreads)
    ge2 = sum(1 for x in spreads if x >= 2.0)
    return {
        "n": n,
        "mean_spread": round(sum(spreads) / n, 6),
        "frac_spread_ge_2": round(ge2 / n, 6),
    }


def main() -> None:
    burst_ok = load_burst_flag()
    # bucket -> list of spr53 at that (day, ts)
    tight_burst: list[float] = []
    tight_no_burst: list[float] = []
    loose: list[float] = []
    per_day: dict[int, dict[str, list[float]]] = {
        d: {"tight_burst": [], "tight_no_burst": [], "loose": []} for d in DAYS
    }

    for day in DAYS:
        sp52: dict[int, float] = {}
        sp53: dict[int, float] = {}
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

        for ts in sorted(sp53.keys()):
            if ts not in sp52:
                continue
            s53 = sp53[ts]
            tight = sp52[ts] <= TH and s53 <= TH
            br = burst_ok.get((day, ts), False)
            if tight and br:
                tight_burst.append(s53)
                per_day[day]["tight_burst"].append(s53)
            elif tight:
                tight_no_burst.append(s53)
                per_day[day]["tight_no_burst"].append(s53)
            else:
                loose.append(s53)
                per_day[day]["loose"].append(s53)

    out = {
        "TH": TH,
        "BURST_MIN": BURST_MIN,
        "pooled": {
            "tight_and_burst": summarize(tight_burst),
            "tight_not_burst": summarize(tight_no_burst),
            "not_tight": summarize(loose),
        },
        "per_day": {
            str(d): {
                "tight_and_burst": summarize(per_day[d]["tight_burst"]),
                "tight_not_burst": summarize(per_day[d]["tight_no_burst"]),
                "not_tight": summarize(per_day[d]["loose"]),
            }
            for d in DAYS
        },
        "interpretation": "If tight_not_burst keeps frac_spread_ge_2 high, ask-1 style passive exits remain placeable after burst ends while Sonic gate still on.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
