#!/usr/bin/env python3
"""
VELVETFRUIT_EXTRACT tape trades: aggressor vs concurrent BBO, joint-tight at trade ts,
forward extract mid deltas K in {5,20,100}.

Phase-1 snap_at / forward_mid; joint_tight from 5200+5300 L1 spread <=2.
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
K_LIST = (5, 20, 100)
EXTRACT = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
TH = 2


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


def joint_tight(day_sym: dict[str, list[Snap]], ts: int) -> bool | None:
    a = snap_at(day_sym.get(S5200, []), ts)
    b = snap_at(day_sym.get(S5300, []), ts)
    if a is None or b is None:
        return None
    return a.spread <= TH and b.spread <= TH


def aggressor_role(px: int, sn: Snap) -> str:
    if px >= sn.ask:
        return "buyer_agg"
    if px <= sn.bid:
        return "seller_agg"
    return "ambig"


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
    return {"n": len(xs), "mean": m, "std": v**0.5}


def main() -> None:
    by_day = {d: load_prices(d) for d in DAYS}
    # key: (role, tight_flag 0/1, K) -> list
    pooled: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    per_day: dict[int, dict[tuple[str, int, int], list[float]]] = {
        d: defaultdict(list) for d in DAYS
    }
    n_skip = 0

    for d in DAYS:
        ser_e = by_day[d].get(EXTRACT, [])
        if not ser_e:
            continue
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["symbol"] != EXTRACT:
                    continue
                ts = int(r["timestamp"])
                price = float(r["price"])
                px = int(round(price))
                sn = snap_at(ser_e, ts)
                if sn is None:
                    n_skip += 1
                    continue
                jt = joint_tight(by_day[d], ts)
                if jt is None:
                    n_skip += 1
                    continue
                tight = 1 if jt else 0
                role = aggressor_role(px, sn)
                for K in K_LIST:
                    fm = forward_mid(ser_e, ts, K)
                    if fm is None:
                        continue
                    delta = fm - sn.mid
                    key = (role, tight, K)
                    pooled[key].append(delta)
                    per_day[d][key].append(delta)

    out: dict = {
        "method": "extract trades only; aggressor from price vs extract snap bid/ask; tight=5200&5300<=2",
        "n_skipped": n_skip,
        "pooled": {},
        "per_day": {},
    }
    for role in ("buyer_agg", "seller_agg", "ambig"):
        out["pooled"][role] = {}
        for tight in (0, 1):
            out["pooled"][role][f"tight_{tight}"] = {
                str(K): summarize(pooled.get((role, tight, K), [])) for K in K_LIST
            }

    out["per_day"] = {}
    for d in DAYS:
        out["per_day"][str(d)] = {}
        for role in ("buyer_agg", "seller_agg", "ambig"):
            out["per_day"][str(d)][role] = {}
            for tight in (0, 1):
                pd = per_day[d]
                out["per_day"][str(d)][role][f"tight_{tight}"] = {
                    str(K): summarize(pd.get((role, tight, K), [])) for K in K_LIST
                }

    pth = OUT / "r4_extract_aggressor_gate_markout.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
