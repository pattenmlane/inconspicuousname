#!/usr/bin/env python3
"""One-off tape stats for VEV_4000 / VEV_4500 (spread + BS IV from mids)."""
import csv
import json
import math
from statistics import NormalDist

_N = NormalDist()
STRIKE = {"VEV_4000": 4000, "VEV_4500": 4500}
VEVS = list(STRIKE)


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return S * _N.cdf(d1) - K * _N.cdf(d2)


def implied_vol(S: float, K: float, T: float, price: float) -> float | None:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-9:
        return None
    lo, hi = 1e-6, 5.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        p = bs_call_price(S, K, T, m)
        if p > price:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return _N.cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    """Call vega: ∂C/∂σ (per 1.0 vol), same sign as standard BS."""
    if T <= 0 or sigma <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return S * _N.pdf(d1) * math.sqrt(T)


def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return _N.pdf(d1) / (S * v)


def main():
    out = {}
    for day in (0, 1, 2):
        path = f"Prosperity4Data/ROUND_3/prices_round_3_day_{day}.csv"
        # TTE mapping: round3description — 7-day option clock; Round 3 sim start TTE=5d.
        # Historical tape day index 0,1,2 → TTE = 7, 6, 5 calendar days (same offset as example: day+1 → TTE 8 in their notation).
        tte_days = 7 - day
        T = tte_days / 365.0
        by_ts: dict[int, dict] = {}
        extract_mid: dict[int, float] = {}
        with open(path, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                ts = int(row["timestamp"])
                prod = row["product"]
                bp = float(row["bid_price_1"] or 0)
                ap = float(row["ask_price_1"] or 0)
                if bp <= 0 or ap <= 0:
                    continue
                mid = (bp + ap) / 2
                sp = ap - bp
                if prod == "VELVETFRUIT_EXTRACT":
                    extract_mid[ts] = mid
                elif prod in STRIKE:
                    if ts not in by_ts:
                        by_ts[ts] = {}
                    by_ts[ts][prod] = {"mid": mid, "spread": sp}

        day_stats = {}
        for v in VEVS:
            spreads = []
            ivs = []
            deltas = []
            vegas = []
            gammas = []
            K = STRIKE[v]
            for ts, d in by_ts.items():
                if v not in d:
                    continue
                S = extract_mid.get(ts)
                if S is None:
                    continue
                spreads.append(d[v]["spread"])
                iv = implied_vol(S, K, T, d[v]["mid"])
                if iv is not None:
                    ivs.append(iv)
                    deltas.append(bs_delta(S, K, T, iv))
                    vegas.append(bs_vega(S, K, T, iv))
                    gammas.append(bs_gamma(S, K, T, iv))
            day_stats[v] = {
                "spread_mean": sum(spreads) / len(spreads) if spreads else None,
                "spread_median": sorted(spreads)[len(spreads) // 2] if spreads else None,
                "iv_mean": sum(ivs) / len(ivs) if ivs else None,
                "iv_std": (
                    math.sqrt(sum((x - sum(ivs) / len(ivs)) ** 2 for x in ivs) / len(ivs))
                    if len(ivs) > 1
                    else None
                ),
                "delta_mean": sum(deltas) / len(deltas) if deltas else None,
                "vega_mean": sum(vegas) / len(vegas) if vegas else None,
                "gamma_mean": sum(gammas) / len(gammas) if gammas else None,
                "n_ticks": len(spreads),
            }
        out[f"day_{day}"] = {"TTE_calendar_days": tte_days, "T_years": T, "per_vev": day_stats}
    with open(
        "manual_traders/R3_VEV/r3v_wide_book_passive_11/tape_vev4000_4500_stats.json",
        "w",
    ) as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
