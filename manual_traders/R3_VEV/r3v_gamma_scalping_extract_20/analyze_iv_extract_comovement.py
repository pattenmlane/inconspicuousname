#!/usr/bin/env python3
"""
Correlation of 1-tick changes: BS implied vol (nearest strike to S) vs extract mid.

TTE: tape day d -> 8-d TTE days (round3work/round3description pattern).
IV from bisection on VEV mid vs extract mid. step=100 between timestamps to limit rows.
Output: analysis_outputs/iv_deltaS_corr.json
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
UNDER = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
TTE_DAYS = {0: 8, 1: 7, 2: 6}
D365 = 365.0
R = 0.0

_N = NormalDist()


def cdf(x: float) -> float:
    return _N.cdf(x)


def bs_call(S: float, K: float, T: float, s: float) -> float:
    if T <= 0 or s <= 0:
        return max(S - K, 0.0)
    st = s * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * s * s) * T) / st
    d2 = d1 - st
    return S * cdf(d1) - K * math.exp(-R * T) * cdf(d2)


def implied_vol(m: float, S: float, K: float, T: float) -> float | None:
    if m <= 0 or m + 1e-9 < max(S - K, 0.0):
        return None
    lo, hi = 1e-5, 4.0
    for _ in range(50):
        s = 0.5 * (lo + hi)
        if bs_call(S, K, T, s) > m:
            hi = s
        else:
            lo = s
    return 0.5 * (lo + hi)


def nearest_k(S: float) -> int:
    return min(STRIKES, key=lambda k: abs(float(k) - S))


def load_series(path: Path) -> list[tuple[int, float, float, float]]:
    """timestamp order, (ts, S, VEV_mid, iv)."""
    by_ts: dict[int, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            by_ts[int(row["timestamp"])][row["product"]] = float(row["mid_price"])
    parts = path.stem.split("_")
    tape_day = int(parts[-1])
    tte = TTE_DAYS.get(tape_day, 6)
    T = tte / D365
    out: list[tuple[int, float, float, float]] = []
    for ts in sorted(by_ts):
        row = by_ts[ts]
        S = row.get(UNDER)
        if S is None:
            continue
        k = nearest_k(S)
        sym = f"VEV_{k}"
        m = row.get(sym)
        if m is None:
            continue
        iv = implied_vol(m, S, float(k), T)
        if iv is None:
            continue
        out.append((ts, S, m, iv))
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def main() -> None:
    step = 100
    by_day: dict = {}
    for d in (0, 1, 2):
        path = _TAPES / f"prices_round_3_day_{d}.csv"
        series = load_series(path)[::step]
        dS: list[float] = []
        dIv: list[float] = []
        for i in range(1, len(series)):
            s0, s1 = series[i - 1][1], series[i][1]
            v0, v1 = series[i - 1][3], series[i][3]
            dS.append(s1 - s0)
            dIv.append(v1 - v0)
        by_day[str(d)] = {
            "n_pairs": len(dS),
            "corr_dIv_dS": pearson(dIv, dS),
            "mean_abs_dS": float(sum(abs(x) for x in dS) / len(dS)) if dS else None,
        }

    payload = {
        "method": "Subsample every 100th timestamp; dIV and dS are successive differences in that subseries.",
        "tte": "8,7,6 days for tape day 0,1,2",
        "by_tape_day": by_day,
    }
    _OUT.mkdir(parents=True, exist_ok=True)
    p = _OUT / "iv_deltaS_corr.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(p.read_text()[: 800])


if __name__ == "__main__":
    main()
