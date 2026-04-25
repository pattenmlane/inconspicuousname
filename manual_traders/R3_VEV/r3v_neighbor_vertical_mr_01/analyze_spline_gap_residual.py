#!/usr/bin/env python3
"""Tape summary: implied vol second-difference (curvature) vs log-strike; gap second-difference.

TTE: CSV day 0/1/2 -> TTE_days = 8 - day (round3work/round3description.txt). IV: BS r=0 from mid.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

OUT = Path(__file__).resolve().parent / "analysis_smile_spline_gaps.json"
DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_3"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMS = [f"VEV_{k}" for k in STRIKES]
UNDER = "VELVETFRUIT_EXTRACT"
N = NormalDist()


def tte_years(csv_day: int) -> float:
    tte_d = 8 - int(csv_day)
    return max(tte_d, 1) / 365.0


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


def load_prices(path: Path):
    by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            by_ts[int(row["timestamp"])][row["product"]] = row
    return by_ts


def main():
    d2_iv: list[float] = []
    d2_gaps: list[float] = []
    for csv_d in (0, 1, 2):
        T = tte_years(csv_d)
        p = load_prices(DATA / f"prices_round_3_day_{csv_d}.csv")
        for ts in sorted(p.keys())[:4000:20]:
            row = p[ts]
            if UNDER not in row:
                continue
            try:
                Su = float(row[UNDER]["mid_price"])
            except (KeyError, ValueError):
                continue
            if Su <= 0:
                continue
            mids: list[float] = []
            for s in SYMS:
                if s not in row:
                    mids = None
                    break
                try:
                    mids.append(float(row[s]["mid_price"]))
                except (KeyError, ValueError):
                    mids = None
                    break
            if mids is None or len(mids) != 10:
                continue
            gaps = [mids[i] - mids[i + 1] for i in range(9)]
            for j in range(1, 8):
                d2g = gaps[j - 1] - 2 * gaps[j] + gaps[j + 1]
                d2_gaps.append(d2g)
            ivs: list[float] = []
            for m, k in zip(mids, STRIKES):
                ivv = implied_vol(m, Su, float(k), T)
                if ivv is None:
                    ivs = None
                    break
                ivs.append(ivv)
            if ivs is None:
                continue
            for j in range(1, 9):
                d2 = ivs[j - 1] - 2 * ivs[j] + ivs[j + 1]
                d2_iv.append(d2)

    def pool(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)
        return {"n": len(xs), "mean": m, "std": math.sqrt(v)}

    doc = {
        "method": "IV from BS call vs mid, T=TTE/365, r=0. Second diff along strike index for interior triples: IV and neighbor gaps m_i - m_{i+1}.",
        "tte": "TTE_days = 8 - csv_day for csv_day in {0,1,2} (same mapping as prior analysis and trader day_idx).",
        "iv_second_diff_interior": pool(d2_iv),
        "gap_second_diff_interior": pool(d2_gaps),
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
