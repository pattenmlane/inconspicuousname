#!/usr/bin/env python3
"""
Lead–lag: at each Mark01→Mark22 VEV trade timestamp, extract forward mid K=20 from that
clock (Phase-1), vs whether an EXTRACT trade occurs strictly later within window W
(same csv day). Windows in raw timestamp units (tape scale).

Also: among events with soon extract, top extract (buyer->seller) pairs (counts).
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
BUYER = "Mark 01"
SELLER = "Mark 22"
WINDOWS = (200, 1000, 5000, 20000, 100000)


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
    ext_ser = {d: by_day[d][EXTRACT] for d in DAYS}

    # day -> sorted unique extract trade timestamps + list of (ts, pair) for first hit
    extract_ts: dict[int, list[int]] = {}
    extract_pairs_at: dict[int, dict[int, str]] = defaultdict(dict)
    for d in DAYS:
        tss: list[int] = []
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                if r["symbol"] != EXTRACT:
                    continue
                ts = int(r["timestamp"])
                b = (r.get("buyer") or "").strip()
                s = (r.get("seller") or "").strip()
                pair = f"{b}->{s}" if b and s else ""
                if ts not in extract_pairs_at[d]:
                    extract_pairs_at[d][ts] = pair
                    tss.append(ts)
        tss = sorted(set(tss))
        extract_ts[d] = tss

    def next_extract_after(d: int, ts: int) -> int | None:
        arr = extract_ts[d]
        if not arr:
            return None
        j = bisect.bisect_right(arr, ts)
        if j >= len(arr):
            return None
        return arr[j]

    # Collect M01->M22 VEV events with fwd20
    events: list[tuple[int, int, float, int | None]] = []
    # (day, ts_vev, fwd20, next_extract_ts or None)
    for d in DAYS:
        ser_e = ext_ser[d]
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
                fm = forward_mid(ser_e, ts, K)
                if fm is None:
                    continue
                nxt = next_extract_after(d, ts)
                events.append((d, ts, fm - sn.mid, nxt))

    out: dict = {
        "n_m01_m22_vev_with_extract_fwd20": len(events),
        "by_window": {},
    }

    for W in WINDOWS:
        soon: list[float] = []
        late: list[float] = []
        for d, ts, fwd, nxt in events:
            if nxt is None or nxt - ts > W:
                late.append(fwd)
            else:
                soon.append(fwd)

        def sm(xs: list[float]) -> dict:
            if not xs:
                return {"n": 0}
            return {"n": len(xs), "mean": statistics.mean(xs)}

        out["by_window"][str(W)] = {
            "extract_trade_within_(ts_vev, ts_vev+W]": sm(soon),
            "no_extract_or_outside_window": sm(late),
        }

    W_pair = 100000
    pair_ctr: Counter[str] = Counter()
    for d, ts, fwd, nxt in events:
        if nxt is not None and nxt - ts <= W_pair:
            pair = extract_pairs_at[d].get(nxt, "")
            if pair:
                pair_ctr[pair] += 1
    out["top_extract_pairs_first_extract_in_(ts_vev, ts_vev+100000]"] = [
        {"pair": p, "n": c} for p, c in pair_ctr.most_common(12)
    ]
    pth = OUT / "r4_m01_m22_vev_extract_leadlag_fwd20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
