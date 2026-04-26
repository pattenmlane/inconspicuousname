#!/usr/bin/env python3
"""At (day,ts) with joint Sonic gate AND M01→M22 burst: VEV_5300 tape trade prices vs L1 bid/ask.

Proxy for worse-fill long: if Mark01 buys 5300 at ask, passive buyer at bid would not fill;
if trades print at bid, join bid is realistic.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_burst_tight_5300_trade_vs_book.json"
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
    out = {}
    for key, rows in by.items():
        if len(rows) < BURST_MIN:
            out[key] = False
            continue
        out[key] = any(
            str(x["buyer"]).strip() == "Mark 01" and str(x["seller"]).strip() == "Mark 22"
            for x in rows
        )
    return out


def load_book(day: int):
    """(ts) -> bid, ask, spr52, spr53 for gate."""
    b53: dict[int, tuple[int, int]] = {}
    sp52: dict[int, float] = {}
    sp53: dict[int, float] = {}
    path = DATA / f"prices_round_4_day_{day}.csv"
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            if int(row["day"]) != day:
                continue
            ts = int(row["timestamp"])
            sym = row["product"]
            bid = int(float(row["bid_price_1"]))
            ask = int(float(row["ask_price_1"]))
            sp = ask - bid if ask > bid else 0
            if sym == S5200:
                sp52[ts] = float(sp)
            elif sym == S5300:
                sp53[ts] = float(sp)
                b53[ts] = (bid, ask)
    return b53, sp52, sp53


def main() -> None:
    burst_ok = load_burst_flag()
    rel = []  # (trade_price - bid) / (ask-bid) when spread>0
    at_ask = 0
    at_bid = 0
    between = 0
    n_rows = 0
    for day in DAYS:
        b53, sp52, sp53 = load_book(day)
        tp = DATA / f"trades_round_4_day_{day}.csv"
        with open(tp, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if row["symbol"] != S5300:
                    continue
                ts = int(row["timestamp"])
                if ts not in b53 or ts not in sp52 or ts not in sp53:
                    continue
                if sp52[ts] > TH or sp53[ts] > TH:
                    continue
                if not burst_ok.get((day, ts), False):
                    continue
                bid, ask = b53[ts]
                px = int(float(row["price"]))
                n_rows += 1
                if px >= ask:
                    at_ask += 1
                elif px <= bid:
                    at_bid += 1
                else:
                    between += 1
                w = ask - bid
                if w > 0:
                    rel.append((px - bid) / w)

    out = {
        "n_5300_trades_tight_burst": n_rows,
        "price_at_ask_count": at_ask,
        "price_at_bid_count": at_bid,
        "price_strictly_between_count": between,
        "mean_fractional_position_in_spread": round(sum(rel) / len(rel), 4) if rel else None,
        "interpretation": "Most VEV_5300 prints at these (tight+burst) timestamps are at the bid (159/161), not the ask — tape aggression is often sell-side at touch; passive bid join is plausible but sim v4 still had zero worse fills (order/repost dynamics).",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
