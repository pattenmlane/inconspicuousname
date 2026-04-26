#!/usr/bin/env python3
"""
Burst timestamps: >=3 trades at same (day, timestamp) counting ALL symbols.
At each burst ts: joint_tight flag from 5200/5300; extract forward mid K=20 from burst ts.

Compare: burst+tight vs burst+loose vs non-burst control (sample of non-burst timestamps
with extract book, same count as bursts per day for balance).
"""
from __future__ import annotations

import bisect
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
K = 20
EXTRACT = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
TH = 2
RNG_SEED = 42


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


def mean(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    m = sum(xs) / len(xs)
    return {"n": len(xs), "mean": m}


def main() -> None:
    rng = random.Random(RNG_SEED)
    by_day = {d: load_prices(d) for d in DAYS}

    burst_ts: dict[int, set[int]] = {d: set() for d in DAYS}
    for d in DAYS:
        cnt: dict[int, int] = defaultdict(int)
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                cnt[int(r["timestamp"])] += 1
        for ts, c in cnt.items():
            if c >= 3:
                burst_ts[d].add(ts)

    fwd_burst_tight: list[float] = []
    fwd_burst_loose: list[float] = []
    fwd_ctrl: list[float] = []
    per_day: dict = {str(d): {"burst_tight": [], "burst_loose": [], "control": []} for d in DAYS}

    for d in DAYS:
        ser_e = by_day[d].get(EXTRACT, [])
        if not ser_e:
            continue
        tss_e = [s.ts for s in ser_e]
        burst_list = sorted(burst_ts[d])
        # control: timestamps with extract snap that are not burst, sample n=len(burst_list)
        non_burst = [t for t in tss_e if t not in burst_ts[d]]
        n_b = len(burst_list)
        ctrl_pick = non_burst[:] if len(non_burst) <= n_b else rng.sample(non_burst, n_b)

        for ts in burst_list:
            sn = snap_at(ser_e, ts)
            if sn is None:
                continue
            fm = forward_mid(ser_e, ts, K)
            if fm is None:
                continue
            delta = fm - sn.mid
            jt = joint_tight(by_day[d], ts)
            if jt is None:
                continue
            if jt:
                fwd_burst_tight.append(delta)
                per_day[str(d)]["burst_tight"].append(delta)
            else:
                fwd_burst_loose.append(delta)
                per_day[str(d)]["burst_loose"].append(delta)

        for ts in ctrl_pick:
            sn = snap_at(ser_e, ts)
            if sn is None:
                continue
            fm = forward_mid(ser_e, ts, K)
            if fm is None:
                continue
            delta = fm - sn.mid
            fwd_ctrl.append(delta)
            per_day[str(d)]["control"].append(delta)

    out = {
        "burst_def": ">=3 trades same (day,timestamp) any symbol",
        "K": K,
        "n_burst_timestamps_by_day": {str(d): len(burst_ts[d]) for d in DAYS},
        "pooled": {
            "burst_joint_tight": mean(fwd_burst_tight),
            "burst_joint_loose": mean(fwd_burst_loose),
            "control_non_burst_sampled": mean(fwd_ctrl),
        },
        "per_day": {k: {kk: mean(vv) for kk, vv in v.items()} for k, v in per_day.items()},
    }
    pth = OUT / "r4_burst_all_syms_gate_extract_fwd20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
