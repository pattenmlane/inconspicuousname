#!/usr/bin/env python3
"""Spearman correlation: VELVETFRUIT_EXTRACT mid vs median IV at strikes closest to spot (Round 3)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "iv_vs_extract_spearman.json"

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)


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


def nearest_strike_iv_row(row: pd.Series, t: float) -> float | None:
    s = float(row["VELVETFRUIT_EXTRACT"])
    ivs: list[float] = []
    for k in STRIKES:
        sym = f"VEV_{k}"
        if sym not in row or pd.isna(row[sym]):
            continue
        iv = iv_bisect(float(row[sym]), s, float(k), t)
        if iv is not None and iv == iv:
            dist = abs(float(k) - s)
            ivs.append((dist, iv))
    if not ivs:
        return None
    ivs.sort(key=lambda x: x[0])
    return float(np.median([x[1] for x in ivs[:3]]))


def main() -> None:
    out: dict = {
        "method": "Per timestamp: S=extract mid, T=dte_eff/365. Median IV of three strikes with smallest |K-S| among those with valid bisection IV. Spearman rho between S and that IV series, subsampled.",
    }
    for day in (0, 1, 2):
        df = pd.read_csv(
            REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv",
            sep=";",
        )
        pvt = df.pivot_table(
            index="timestamp", columns="product", values="mid_price", aggfunc="first"
        )
        idx = pvt.index.to_numpy()
        rng = np.random.default_rng(7 + day)
        sample = sorted(rng.choice(idx, size=min(2000, len(idx)), replace=False).tolist())
        s_list: list[float] = []
        iv_list: list[float] = []
        for ts in sample:
            row = pvt.loc[ts]
            t = dte_eff(day, int(ts)) / 365.0
            ivm = nearest_strike_iv_row(row, t)
            if ivm is None:
                continue
            s_list.append(float(row["VELVETFRUIT_EXTRACT"]))
            iv_list.append(ivm)
        if len(s_list) < 10:
            out[str(day)] = {"n": len(s_list), "spearman": None, "pvalue": None}
            continue
        r, p = spearmanr(s_list, iv_list)
        out[str(day)] = {"n": len(s_list), "spearman": float(r), "pvalue": float(p)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
