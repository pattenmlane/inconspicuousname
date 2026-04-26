#!/usr/bin/env python3
"""
Phase 1 graph extension: **adjacent** trade pairs (same csv day, sorted by ts then sym)
where middle name matches: trade_i seller == trade_{i+1} buyer (A->B->C flow).

At the **second** trade timestamp, extract forward mid K=20 (Phase-1 conventions).
Summarize top motifs by count; compare pooled mean to **all** second-leg timestamps
(any adjacent pair without the middle match — control).
"""
from __future__ import annotations

import bisect
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
K = 20
EXTRACT = "VELVETFRUIT_EXTRACT"


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


def load_day_trades(day: int) -> list[dict]:
    path = DATA / f"trades_round_4_day_{day}.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            rows.append(
                {
                    "ts": int(r["timestamp"]),
                    "buyer": (r.get("buyer") or "").strip(),
                    "seller": (r.get("seller") or "").strip(),
                    "sym": r["symbol"],
                }
            )
    rows.sort(key=lambda x: (x["ts"], x["sym"]))
    return rows


def sm(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": statistics.mean(xs), "median": statistics.median(xs)}


def main() -> None:
    by_day_prices = {d: load_prices(d) for d in DAYS}
    ext = {d: by_day_prices[d][EXTRACT] for d in DAYS}

    motif_fwd: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    control_fwd: list[float] = []
    n_pairs = 0
    n_motif = 0

    for d in DAYS:
        trs = load_day_trades(d)
        ser_e = ext[d]
        for i in range(len(trs) - 1):
            a, b = trs[i], trs[i + 1]
            n_pairs += 1
            ts2 = b["ts"]
            sn = snap_at(ser_e, ts2)
            if sn is None:
                continue
            fm = forward_mid(ser_e, ts2, K)
            if fm is None:
                continue
            delta = fm - sn.mid

            if a["seller"] and b["buyer"] and a["seller"] == b["buyer"]:
                key = (a["buyer"], a["seller"], b["seller"])
                motif_fwd[key].append(delta)
                n_motif += 1
            else:
                control_fwd.append(delta)

    ctr = Counter({k: len(v) for k, v in motif_fwd.items()})
    top = ctr.most_common(20)
    top_detail = []
    for (trip, c) in top:
        xs = motif_fwd[trip]
        top_detail.append(
            {
                "motif": f"{trip[0]}->{trip[1]}->{trip[2]}",
                "n": c,
                **sm(xs),
            }
        )

    out = {
        "definition": "Adjacent rows in day trade CSV after sort(ts,sym); motif if trade[i].seller==trade[i+1].buyer; fwd20 extract at ts of trade i+1",
        "n_adjacent_pairs": n_pairs,
        "n_pairs_with_middle_match": n_motif,
        "pooled_motif_extract_fwd20": sm([x for v in motif_fwd.values() for x in v]),
        "pooled_control_extract_fwd20": sm(control_fwd),
        "top_motifs": top_detail,
    }
    pth = OUT / "r4_twohop_adjacent_extract_fwd20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
