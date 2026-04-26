#!/usr/bin/env python3
"""Per-VEV Black-Scholes vega (r=0) on ROUND_3 price tapes, days 0-2, subsampled rows.

S: VELVETFRUIT_EXTRACT mid; K: strike; T: 8/365, 7/365, 6/365 by csv day; IV from call mid.
Output: mean vega by symbol to compare scale across strike (Greek / IV thread).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

N = NormalDist()
ROOT = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"
DAYS = (0, 1, 2)
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMS = [f"VEV_{k}" for k in STRIKES]
UNDER = "VELVETFRUIT_EXTRACT"
EVERY = 10  # subsample every Nth global row with full book


def tte_years(day_tag: int) -> float:
    return max(8 - max(0, min(day_tag, 2)), 1) / 365.0


def half_spread(bid1: str, ask1: str) -> bool:
    try:
        b, a = int(bid1), int(ask1)
    except (TypeError, ValueError):
        return False
    return a > b


def bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0 or S <= 0:
        return max(S - K, 0.0)
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * st)
    d2 = d1 - sig * st
    return S * N.cdf(d1) - K * N.cdf(d2)


def implied_vol(mid: float, S: float, K: float, T: float) -> float | None:
    ins = max(S - K, 0.0)
    if mid <= ins + 1e-9 or T <= 0:
        return None
    lo, hi = 1e-4, 4.0
    for _ in range(40):
        m = 0.5 * (lo + hi)
        p = bs_call(S, K, T, m)
        if p > mid:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def bs_vega(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0 or S <= 0:
        return 0.0
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * st)
    return S * st * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)


def main() -> None:
    by_sym: dict[str, list[float]] = defaultdict(list)
    n_ok = 0
    for d in DAYS:
        T = tte_years(d)
        path = ROOT / f"prices_round_3_day_{d}.csv"
        rows_by_ts: dict[int, dict[str, dict]] = {}
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != d:
                    continue
                ts = int(row["timestamp"])
                p = row["product"]
                if p not in SYMS and p != UNDER:
                    continue
                if ts not in rows_by_ts:
                    rows_by_ts[ts] = {}
                rows_by_ts[ts][p] = row
        for ts in sorted(rows_by_ts):
            m = rows_by_ts[ts]
            if UNDER not in m:
                continue
            if not all(x in m for x in SYMS):
                continue
            ru = m[UNDER]
            if not half_spread(ru["bid_price_1"], ru["ask_price_1"]):
                continue
            n_ok += 1
            if (n_ok % EVERY) != 0:
                continue
            bu, au = int(ru["bid_price_1"]), int(ru["ask_price_1"])
            S = 0.5 * (bu + au)
            for sym, K in zip(SYMS, STRIKES):
                ro = m[sym]
                if not half_spread(ro["bid_price_1"], ro["ask_price_1"]):
                    continue
                b1, a1 = int(ro["bid_price_1"]), int(ro["ask_price_1"])
                midv = 0.5 * (b1 + a1)
                iv = implied_vol(float(midv), S, float(K), T)
                if iv is None:
                    continue
                vg = bs_vega(S, float(K), T, iv)
                by_sym[sym].append(vg)

    out: dict = {"method": "BS vega with IV from mid, T from round3 8,7,6d mapping", "subsample": f"1/{EVERY} full-timestamp rows"}
    out["by_symbol"] = {}
    for s in SYMS:
        arr = by_sym.get(s) or []
        if not arr:
            out["by_symbol"][s] = {"n": 0, "mean_vega": 0.0}
        else:
            out["by_symbol"][s] = {
                "n": len(arr),
                "mean_vega": sum(arr) / len(arr),
            }
    ranked: list[tuple[str, float]] = []
    for s in SYMS:
        v = out["by_symbol"][s].get("mean_vega", 0)
        ranked.append((s, v))
    ranked.sort(key=lambda x: -x[1])
    out["rank_by_mean_vega"] = ranked
    out["note"] = "Use vega in edge score: normalize cross-strike z by vol sensitivity, or boost strikes where vega*beta/spread is high."
    Path(__file__).resolve().parent.joinpath("analysis_vega_by_strike_r3.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote analysis_vega_by_strike_r3.json")


if __name__ == "__main__":
    main()
