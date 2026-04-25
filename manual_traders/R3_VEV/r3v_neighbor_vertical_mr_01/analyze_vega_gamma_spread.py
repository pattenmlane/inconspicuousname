#!/usr/bin/env python3
"""Pooled book + Greek snapshot on Round 3 tapes: half-spread and BS gamma (near-ATM VEV)."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

N = NormalDist()
_DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"
_OUT = Path(__file__).resolve().parent / "analysis_vega_gamma_spread.json"
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
    for _ in range(40):
        m = 0.5 * (lo + hi)
        p = bs_call(S, K, T, m)
        if p > mid:
            hi = m
        else:
            lo = m
    s = 0.5 * (lo + hi)
    return s if s > 1e-4 else None


def bs_gamma_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0 or S <= 0:
        return 0.0
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * st)
    n1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return n1 / (S * st * sig)


def load(path: Path):
    d: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open() as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            d[int(row["timestamp"])][row["product"]] = row
    return d


def half_spread(row: dict) -> float | None:
    try:
        bb = int(row["bid_price_1"])
        ba = int(row["ask_price_1"])
        return 0.5 * (ba - bb)
    except (KeyError, TypeError, ValueError):
        return None


def main():
    half_by_sym = {s: [] for s in SYMS}
    gams: list[float] = []
    for d in (0, 1, 2):
        T = tte_years(d)
        p = load(_DATA / f"prices_round_3_day_{d}.csv")
        for ts in sorted(p.keys())[:5000:25]:
            row = p[ts]
            if UNDER not in row:
                continue
            try:
                Su = float(row[UNDER]["mid_price"])
            except (KeyError, ValueError):
                continue
            j = min(range(10), key=lambda k: abs(STRIKES[k] - Su))
            sym = SYMS[j]
            K = float(STRIKES[j])
            if sym not in row:
                continue
            mid_c = None
            try:
                mid_c = float(row[sym]["mid_price"])
            except (KeyError, ValueError):
                pass
            hs = half_spread(row[sym])
            if hs is not None:
                half_by_sym[sym].append(hs)
            if mid_c is not None:
                iv = implied_vol(mid_c, Su, K, T)
                if iv is not None:
                    gams.append(bs_gamma_call(Su, K, T, iv))

    def stat(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
        return {"n": len(xs), "mean": m, "std": math.sqrt(v)}

    out = {
        "tte": "TTE_days = 8 - csv_day for day 0/1/2 files.",
        "half_spread_by_vev_pooled": {s: stat(half_by_sym[s]) for s in SYMS},
        "bs_gamma_at_nearest_strike": stat(gams),
        "method": "IV from mid via BS bisection (r=0). Gamma: Black-Scholes call d^2C/dS^2, strike nearest to extract mid at each sample.",
    }
    _OUT.write_text(json.dumps(out, indent=2))
    print("Wrote", _OUT)


if __name__ == "__main__":
    main()
