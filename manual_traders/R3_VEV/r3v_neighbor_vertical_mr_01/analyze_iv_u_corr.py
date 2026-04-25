#!/usr/bin/env python3
"""Sample correlation: VELVETFRUIT_EXTRACT mid vs mean BS IV (10 strikes) on ROUND_3 tapes.

T: TTE_days/365 with TTE_days = 8 - csv_day. IV: bisection on European call, r=0.
Outputs analysis_iv_u_mean_corr.json
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

N = NormalDist()
DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_iv_u_mean_corr.json"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMS = [f"VEV_{k}" for k in STRIKES]
UNDER = "VELVETFRUIT_EXTRACT"


def tte_years(csv_day: int) -> float:
    return max(8 - int(csv_day), 1) / 365.0


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
    for _ in range(50):
        m = 0.5 * (lo + hi)
        p = bs_call(S, K, T, m)
        if p > mid:
            hi = m
        else:
            lo = m
    s = 0.5 * (lo + hi)
    return s if s > 1e-4 else None


def load(path: Path):
    d: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open() as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            d[int(row["timestamp"])][row["product"]] = row
    return d


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dxe = sum((a - mx) ** 2 for a in xs) ** 0.5
    dye = sum((b - my) ** 2 for b in ys) ** 0.5
    if dxe < 1e-15 or dye < 1e-15:
        return None
    return num / (dxe * dye)


def main():
    u_list: list[float] = []
    ivm_list: list[float] = []
    for d in (0, 1, 2):
        T = tte_years(d)
        prices = load(DATA / f"prices_round_3_day_{d}.csv")
        for ts in sorted(prices.keys())[:8000:15]:
            row = prices[ts]
            if UNDER not in row:
                continue
            try:
                Su = float(row[UNDER]["mid_price"])
            except (KeyError, ValueError):
                continue
            ivs = []
            for s, k in zip(SYMS, STRIKES):
                if s not in row:
                    continue
                m = float(row[s]["mid_price"])
                iv = implied_vol(m, Su, float(k), T)
                if iv is None:
                    ivs = None
                    break
                ivs.append(iv)
            if ivs is None:
                continue
            u_list.append(Su)
            ivm_list.append(sum(ivs) / len(ivs))
    r = pearson(u_list, ivm_list)
    doc = {
        "method": "Per timestamp: mean of 10 option IVs (BS, r=0) vs VELVETFRUIT_EXTRACT mid; T=TTE/365, TTE_days=8-csv_day.",
        "n_pairs": len(u_list),
        "pearson_u_vs_mean_iv": r,
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print("Wrote", OUT, "r=", r, "n=", len(u_list))


if __name__ == "__main__":
    main()
