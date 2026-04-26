#!/usr/bin/env python3
"""
All VEV_* tape trades: forward mid on the *traded* strike at K in {5,20,100} (Phase-1
snap_at / forward_mid), stratified by Sonic joint-tight (5200+5300 L1<=2) at trade ts.

Welch tight vs loose per strike for K=20 when both n>=25; pooled all-VEV K=20 Welch.
"""
from __future__ import annotations

import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("Prosperity4Data/ROUND_4")

DAYS = (1, 2, 3)
K_LIST = (5, 20, 100)
S5200 = "VEV_5200"
S5300 = "VEV_5300"
TH = 2
MIN_WELCH = 25


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


def welch(a: list[float], b: list[float]) -> dict | None:
    x = np.array(a, dtype=float)
    y = np.array(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return None
    t = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return {
        "mean_tight": float(np.mean(x)),
        "mean_loose": float(np.mean(y)),
        "n_tight": int(len(x)),
        "n_loose": int(len(y)),
        "t_stat": float(t.statistic),
        "p_value": float(t.pvalue),
    }


def main() -> None:
    by_day = {d: load_prices(d) for d in DAYS}
    # (sym, tight01, K) -> list of fwd deltas on traded sym
    cells: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    pooled: dict[tuple[int, int], list[float]] = defaultdict(list)

    for d in DAYS:
        path = DATA / f"trades_round_4_day_{d}.csv"
        with open(path, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                sym = r["symbol"]
                if not sym.startswith("VEV_"):
                    continue
                ts = int(r["timestamp"])
                ser = by_day[d].get(sym)
                if not ser:
                    continue
                sn = snap_at(ser, ts)
                if sn is None:
                    continue
                jt = joint_tight(by_day[d], ts)
                if jt is None:
                    continue
                ti = 1 if jt else 0
                for K in K_LIST:
                    fm = forward_mid(ser, ts, K)
                    if fm is None:
                        continue
                    delta = fm - sn.mid
                    cells[(sym, ti, K)].append(delta)
                    pooled[(ti, K)].append(delta)

    def mean_n(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "mean": float(sum(xs) / len(xs))}

    per_sym_k20: dict[str, dict] = {}
    welch_by_sym: dict[str, dict | None] = {}
    strikes = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
    for sym in strikes:
        t = cells.get((sym, 1, 20), [])
        lo = cells.get((sym, 0, 20), [])
        per_sym_k20[sym] = {"tight": mean_n(t), "loose": mean_n(lo)}
        if len(t) >= MIN_WELCH and len(lo) >= MIN_WELCH:
            welch_by_sym[sym] = welch(t, lo)
        else:
            welch_by_sym[sym] = None

    t_all = pooled.get((1, 20), [])
    lo_all = pooled.get((0, 20), [])

    out = {
        "method": "fwd on traded VEV mid; joint_tight from 5200+5300 at trade ts",
        "per_strike_k20_means": per_sym_k20,
        "per_strike_k20_welch_tight_vs_loose_min_n_25": welch_by_sym,
        "pooled_all_vev_k20": {"tight": mean_n(t_all), "loose": mean_n(lo_all)},
        "pooled_all_vev_k20_welch": welch(t_all, lo_all) if len(t_all) >= 30 and len(lo_all) >= 30 else None,
        "pooled_all_vev_k5": {
            "tight": mean_n(pooled.get((1, 5), [])),
            "loose": mean_n(pooled.get((0, 5), [])),
        },
        "pooled_all_vev_k100": {
            "tight": mean_n(pooled.get((1, 100), [])),
            "loose": mean_n(pooled.get((0, 100), [])),
        },
    }
    pth = OUT / "r4_vev_trade_fwd_joint_gate.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
