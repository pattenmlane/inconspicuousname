#!/usr/bin/env python3
"""
Cluster (day-block) bootstrap for Phase-1 participant screen cells (party, symbol,
buyer_agg/seller_agg/ambig): K=20 extract or traded-symbol fwd, same tape logic as
run_r4_phase1_counterparty.py.

For each of the top-N cells by pooled n from a fresh screen, report pooled mean and
2.5/97.5 percentile of bootstrap mean (2000 resamples: each resample draws 3 days with
replacement and concatenates all deltas from those days).
"""
from __future__ import annotations

import bisect
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
K = 20
RNG_SEED = 7
N_BOOT = 2000
TOP_N = 25


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


def load_trades(day: int) -> list[dict]:
    path = DATA / f"trades_round_4_day_{day}.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append(
                {
                    "day": day,
                    "ts": int(r["timestamp"]),
                    "buyer": (r.get("buyer") or "").strip(),
                    "seller": (r.get("seller") or "").strip(),
                    "sym": r["symbol"],
                    "price": float(r["price"]),
                }
            )
    rows.sort(key=lambda x: (x["ts"], x["sym"]))
    return rows


def main() -> None:
    rng = random.Random(RNG_SEED)
    price_by_day = {d: load_prices(d) for d in DAYS}
    # (party, sym, role) -> day -> list of deltas on traded sym
    by_cell_day: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for d in DAYS:
        for tr in load_trades(d):
            sym = tr["sym"]
            ser = price_by_day.get(d, {}).get(sym)
            if not ser:
                continue
            sn = snap_at(ser, tr["ts"])
            if sn is None:
                continue
            px = int(round(tr["price"]))
            role = "ambig"
            if px >= sn.ask:
                role = "buyer_agg"
            elif px <= sn.bid:
                role = "seller_agg"
            fm = forward_mid(ser, tr["ts"], K)
            if fm is None:
                continue
            delta = fm - sn.mid
            for party in (tr["buyer"], tr["seller"]):
                if not party:
                    continue
                pr = "buy" if party == tr["buyer"] else "sell"
                key = (party, sym, role)
                by_cell_day[key][d].append(delta)

    # pooled n per cell
    pooled_n: dict[tuple[str, str, str], int] = {}
    for key, dd in by_cell_day.items():
        pooled_n[key] = sum(len(v) for v in dd.values())

    top_keys = sorted(pooled_n.keys(), key=lambda k: -pooled_n[k])[:TOP_N]

    day_list = list(DAYS)
    rows_out = []
    for key in top_keys:
        dd = by_cell_day[key]
        flat = [x for d in DAYS for x in dd.get(d, [])]
        if len(flat) < 15:
            continue
        mu = statistics.mean(flat)
        boots: list[float] = []
        for _ in range(N_BOOT):
            samp: list[float] = []
            for _b in range(3):
                pick = rng.choice(day_list)
                samp.extend(dd.get(pick, []))
            if samp:
                boots.append(statistics.mean(samp))
        boots.sort()
        lo = boots[int(0.025 * len(boots))] if boots else None
        hi = boots[int(0.975 * len(boots)) - 1] if len(boots) > 1 else boots[0] if boots else None
        rows_out.append(
            {
                "party": key[0],
                "symbol": key[1],
                "role": key[2],
                "n": len(flat),
                "n_by_day": {str(d): len(dd.get(d, [])) for d in DAYS},
                "pooled_mean_fwd20": mu,
                "bootstrap_mean_ci95": [lo, hi],
            }
        )

    out = {
        "method": "2000 cluster-bootstrap: each draw resamples 3 days with replacement, concatenates deltas, mean",
        "K": K,
        "top_cells_by_pooled_n": rows_out,
    }
    pth = OUT / "r4_participant_screen_fwd20_day_bootstrap.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
