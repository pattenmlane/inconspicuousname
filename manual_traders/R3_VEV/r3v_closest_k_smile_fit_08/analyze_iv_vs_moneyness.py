#!/usr/bin/env python3
"""Median BS implied vol by |log-moneyness| bucket (Round 3 tapes)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "iv_vs_log_moneyness.json"

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s: float, k: float, t: float, sig: float) -> float:
    if t <= 1e-12 or sig <= 1e-12:
        return max(s - k, 0.0)
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / v
    d2 = d1 - v
    return s * norm_cdf(d1) - k * norm_cdf(d2)


def iv_bisect(px: float, s: float, k: float, t: float) -> float | None:
    if px <= max(s - k, 0.0) + 1e-6 or px >= s - 1e-6:
        return None
    lo, hi = 1e-4, 12.0
    if bs_call(s, k, t, lo) - px > 0 or bs_call(s, k, t, hi) - px < 0:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bs_call(s, k, t, mid) >= px:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def dte_eff(day: int, ts: int) -> float:
    return max(8.0 - float(day) - (int(ts) // 100) / 10_000.0, 1e-6)


def main() -> None:
    buckets = (0.0, 0.03, 0.06, 0.10, 0.15, 1.0)
    out: dict = {
        "method": "For each (day,timestamp,VEV), IV from mid via bisection, r=0, T=dte_eff/365 per plot_iv_smile / round3 time map. Bucket by abs(log(K/S)) using mid extract S; report median IV per bucket.",
        "buckets": [f"({buckets[i]:.2f},{buckets[i+1]:.2f}]" for i in range(len(buckets) - 1)],
        "by_day": {},
    }
    for day in (0, 1, 2):
        df = pd.read_csv(
            REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv",
            sep=";",
        )
        pvt = df.pivot_table(
            index="timestamp", columns="product", values="mid_price", aggfunc="first"
        )
        if "VELVETFRUIT_EXTRACT" not in pvt.columns:
            continue
        ts_idx = pvt.index.to_numpy()
        rng = np.random.default_rng(42 + day)
        sample = sorted(rng.choice(ts_idx, size=min(400, len(ts_idx)), replace=False).tolist())
        per_bucket: dict[str, list[float]] = {out["buckets"][i]: [] for i in range(len(buckets) - 1)}

        for ts in sample:
            s = float(pvt.at[ts, "VELVETFRUIT_EXTRACT"])
            t = dte_eff(day, int(ts)) / 365.0
            for k in STRIKES:
                sym = f"VEV_{k}"
                if sym not in pvt.columns or pd.isna(pvt.at[ts, sym]):
                    continue
                px = float(pvt.at[ts, sym])
                iv = iv_bisect(px, s, float(k), t)
                if iv is None or iv != iv:
                    continue
                lm = abs(math.log(k / s))
                for i in range(len(buckets) - 1):
                    if buckets[i] < lm <= buckets[i + 1]:
                        per_bucket[out["buckets"][i]].append(iv)
                        break
        out["by_day"][str(day)] = {
            b: {
                "n": len(vs),
                "median_iv": float(np.median(vs)) if vs else None,
            }
            for b, vs in per_bucket.items()
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
