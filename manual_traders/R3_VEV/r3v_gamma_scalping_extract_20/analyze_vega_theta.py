#!/usr/bin/env python3
"""
BS vega and theta (call) from tape mids; same TTE convention as analyze_tapes_iv_greeks.py.

TTE_days for CSV day index 0,1,2: 8, 7, 6 (round3work/round3description.txt pattern).
r=0, T = TTE_days/365.
Subsample timestamps (step 500) for speed.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

_ROOT = Path(__file__).resolve().parents[3]
_TAPES = _ROOT / "Prosperity4Data" / "ROUND_3"
_OUT = Path(__file__).resolve().parent / "analysis_outputs"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
UNDER = "VELVETFRUIT_EXTRACT"
TTE_DAYS = {0: 8, 1: 7, 2: 6}
DAYS_PER_YEAR = 365.0
R = 0.0

_N = NormalDist()


def cdf(x: float) -> float:
    return _N.cdf(x)


def pdf(x: float) -> float:
    return _N.pdf(x)


def bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    st = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sig * sig) * T) / st
    d2 = d1 - st
    return S * cdf(d1) - K * math.exp(-R * T) * cdf(d2)


def implied_vol(mid: float, S: float, K: float, T: float) -> float | None:
    if mid <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    if mid + 1e-9 < max(S - K, 0.0):
        return None
    lo, hi = 1e-5, 4.0
    for _ in range(55):
        s = 0.5 * (lo + hi)
        if bs_call(S, K, T, s) > mid:
            hi = s
        else:
            lo = s
    return 0.5 * (lo + hi)


def d1d2(S: float, K: float, T: float, sig: float) -> tuple[float, float]:
    st = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sig * sig) * T) / st
    d2 = d1 - st
    return d1, d2


def vega(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return 0.0
    d1, _ = d1d2(S, K, T, sig)
    return S * pdf(d1) * math.sqrt(T)


def theta_call(S: float, K: float, T: float, sig: float) -> float:
    """Theta per year (negative for long option under standard convention)."""
    if T <= 0 or sig <= 0:
        return 0.0
    st = sig * math.sqrt(T)
    d1, d2 = d1d2(S, K, T, sig)
    term1 = -(S * pdf(d1) * sig) / (2.0 * math.sqrt(T))
    term2 = -R * K * math.exp(-R * T) * cdf(d2)
    return term1 + term2


def load_ts(path: Path) -> dict[int, dict[str, float]]:
    by_ts: dict[int, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            by_ts[int(row["timestamp"])][row["product"]] = float(row["mid_price"])
    return by_ts


def two_nearest(S: float) -> tuple[int, int]:
    ranked = sorted(STRIKES, key=lambda k: abs(float(k) - S))
    a, b = ranked[0], ranked[1]
    return (a, b) if a < b else (b, a)


def main() -> None:
    step = 500
    rows_out: list[dict] = []
    agg_vega: dict[int, list[float]] = defaultdict(list)
    agg_theta: dict[int, list[float]] = defaultdict(list)

    for tape_day in (0, 1, 2):
        data = load_ts(_TAPES / f"prices_round_3_day_{tape_day}.csv")
        tte = TTE_DAYS[tape_day]
        T = tte / DAYS_PER_YEAR
        for ts in sorted(data.keys())[::step]:
            mp = data[ts]
            S = mp.get(UNDER)
            if S is None:
                continue
            k1, k2 = two_nearest(S)
            for k in (k1, k2):
                sym = f"VEV_{k}"
                m = mp.get(sym)
                if m is None:
                    continue
                sig = implied_vol(m, S, float(k), T)
                if sig is None:
                    continue
                v = vega(S, float(k), T, sig)
                th = theta_call(S, float(k), T, sig) / 365.0
                rows_out.append(
                    {
                        "tape_day": tape_day,
                        "timestamp": ts,
                        "strike": k,
                        "S": S,
                        "vega": v,
                        "theta_per_day": th,
                    }
                )
                agg_vega[tape_day].append(v)
                agg_theta[tape_day].append(th)

    summary = {
        "methodology": "BS vega and theta (per calendar day) from implied vol solved from mids; TTE 8,7,6 for tape days 0,1,2; step 500.",
        "mean_vega_by_tape_day": {str(d): round(sum(x) / len(x), 6) for d, x in sorted(agg_vega.items()) if x},
        "mean_abs_theta_per_day_by_tape_day": {
            str(d): round(sum(abs(t) for t in x) / len(x), 6) for d, x in sorted(agg_theta.items()) if x
        },
    }

    _OUT.mkdir(parents=True, exist_ok=True)
    p_json = _OUT / "vega_theta_summary.json"
    p_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    p_csv = _OUT / "vega_theta_sample.csv"
    if rows_out:
        with p_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    print(json.dumps({"wrote": [str(p_json), str(p_csv)], "rows": len(rows_out)}))


if __name__ == "__main__":
    main()
