#!/usr/bin/env python3
"""
Two-strike IV 'skew' on Round-3 tapes: at each subsampled timestamp, take two
nearest strikes to extract mid, solve BS IV for each, record IV_hi - IV_lo.

TTE: tape day d -> (8-d) days; T = TTE/365; r=0.
Output: analysis_outputs/two_strike_iv_skew.json
"""
from __future__ import annotations

import csv
import json
import math
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
    for _ in range(55):
        s = 0.5 * (lo + hi)
        if bs_call(S, K, T, s) > m:
            hi = s
        else:
            lo = s
    return 0.5 * (lo + hi)


def two_nearest(S: float) -> tuple[int, int]:
    r = sorted(STRIKES, key=lambda k: abs(float(k) - S))
    a, b = r[0], r[1]
    return (a, b) if a < b else (b, a)


def main() -> None:
    step = 500
    summary: dict = {"method": "step 500; IV from mids; skew = IV(higher K) - IV(lower K) for the two nearest strikes", "by_tape_day": {}}
    for d in (0, 1, 2):
        diffs: list[float] = []
        path = _TAPES / f"prices_round_3_day_{d}.csv"
        T = TTE_DAYS[d] / D365
        by_ts: dict[int, dict[str, float]] = {}
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = int(row["timestamp"])
                if ts not in by_ts:
                    by_ts[ts] = {}
                by_ts[ts][row["product"]] = float(row["mid_price"])
        for ts in sorted(by_ts.keys())[::step]:
            mp = by_ts[ts]
            S = mp.get(UNDER)
            if S is None:
                continue
            k_lo, k_hi = two_nearest(S)
            m_lo = mp.get(f"VEV_{k_lo}")
            m_hi = mp.get(f"VEV_{k_hi}")
            if m_lo is None or m_hi is None:
                continue
            iv_lo = implied_vol(m_lo, S, float(k_lo), T)
            iv_hi = implied_vol(m_hi, S, float(k_hi), T)
            if iv_lo is None or iv_hi is None:
                continue
            # k_lo < k_hi by construction: IV(higher K) - IV(lower K)
            diffs.append(iv_hi - iv_lo)
        if diffs:
            summary["by_tape_day"][str(d)] = {
                "n": len(diffs),
                "mean_skew": sum(diffs) / len(diffs),
                "mean_abs_skew": sum(abs(x) for x in diffs) / len(diffs),
            }
    _OUT.mkdir(parents=True, exist_ok=True)
    p = _OUT / "two_strike_iv_skew.json"
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(p.read_text())


if __name__ == "__main__":
    main()
