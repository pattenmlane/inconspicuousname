#!/usr/bin/env python3
"""Phase 2.5: BS IV smile residual (5200/5300 only) vs Mark 01→22 trade density by hour.

Simple ATM proxy: IV from mid at each timestamp for 5200 and 5300, fit linear IV vs log(K/S)
using extract mid S; residual = avg(|iv5200-iv5300|) or max deviation — here use
abs difference of IVs as "steepness" proxy. Count M01→M22 trades in same hour bin.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

N = NormalDist()
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "r4_phase2_iv_mark_density.json"
DAYS = (1, 2, 3)
STRIKES = (5200, 5300)
TTE = 4 / 365.0


def bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0 or S <= 0:
        return max(S - K, 0.0)
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * st)
    d2 = d1 - sig * st
    return S * N.cdf(d1) - K * N.cdf(d2)


def iv(mid: float, S: float, K: float, T: float) -> float | None:
    ins = max(S - K, 0.0)
    if mid <= ins + 1e-9:
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


def main() -> None:
    bins_iv: dict[int, list[float]] = defaultdict(list)
    bins_cnt: dict[int, int] = defaultdict(int)
    for day in DAYS:
        # load mids per ts
        mids: dict[str, dict[int, float]] = defaultdict(dict)
        path = DATA / f"prices_round_4_day_{day}.csv"
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if int(row["day"]) != day:
                    continue
                sym = row["product"]
                if sym not in ("VELVETFRUIT_EXTRACT", "VEV_5200", "VEV_5300"):
                    continue
                mids[sym][int(row["timestamp"])] = float(row["mid_price"])
        ts_common = sorted(
            set(mids["VEV_5200"]) & set(mids["VEV_5300"]) & set(mids["VELVETFRUIT_EXTRACT"])
        )
        for ts in ts_common[::50]:  # subsample for speed
            S = mids["VELVETFRUIT_EXTRACT"][ts]
            m52 = mids["VEV_5200"][ts]
            m53 = mids["VEV_5300"][ts]
            a = iv(m52, S, 5200.0, TTE)
            b = iv(m53, S, 5300.0, TTE)
            if a is None or b is None:
                continue
            h = ts // 100000
            bins_iv[h].append(abs(a - b))
        tp = DATA / f"trades_round_4_day_{day}.csv"
        with open(tp, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                if row["buyer"] == "Mark 01" and row["seller"] == "Mark 22":
                    ts = int(row["timestamp"])
                    bins_cnt[ts // 100000] += 1

    rows = []
    for h in sorted(set(bins_iv) | set(bins_cnt)):
        if not bins_iv[h]:
            continue
        miv = sum(bins_iv[h]) / len(bins_iv[h])
        rows.append({"hour_bin": h, "mean_abs_iv_diff_5200_5300": round(miv, 6), "m01_m22_trades": bins_cnt[h], "n_iv_samples": len(bins_iv[h])})

    OUT.write_text(json.dumps({"per_hour": rows}, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
