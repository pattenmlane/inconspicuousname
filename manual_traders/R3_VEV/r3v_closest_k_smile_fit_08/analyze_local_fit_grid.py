#!/usr/bin/env python3
"""Compare local-smile fit quality for closest-k in {4,5,6,7} on ROUND_3 tapes.

Metric: median absolute pricing residual in ticks vs observed voucher mids.
IV method: bisection implied vol from mids, r=0, T=dte_eff/365 with
dte_eff(day,ts)=max((8-day) - (ts//100)/10000, 1e-6), consistent with prior analyses.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "local_fit_k_grid.json"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s: float, k: float, t: float, sig: float) -> float:
    if t <= 1e-12 or sig <= 1e-12:
        return max(s - k, 0.0)
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / v
    d2 = d1 - v
    return s * _norm_cdf(d1) - k * _norm_cdf(d2)


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


def iv_surface_value(s: float, k: float, coeffs: tuple[float, float, float] | None) -> float | None:
    if coeffs is None or s <= 0 or k <= 0:
        return None
    a, b, c = coeffs
    x = math.log(k / s)
    sig = a * x * x + b * x + c
    if sig < 1e-4 or sig > 10.0:
        return None
    return sig


def fit_surface(s: float, pairs: list[tuple[float, float, float]], k_closest: int) -> tuple[float, float, float] | None:
    use = sorted(pairs, key=lambda p: p[1])[: min(k_closest, len(pairs))]
    if len(use) < 2:
        return None
    xs = [math.log(p[0] / s) for p in use]
    ys = [p[2] for p in use]
    if len(use) == 2:
        return 0.0, 0.0, float(np.mean(ys))
    coef = np.polyfit(np.array(xs), np.array(ys), deg=2 if len(use) >= 3 else 1)
    if len(coef) == 2:
        b, c = coef
        return 0.0, float(b), float(c)
    a, b, c = coef
    return float(a), float(b), float(c)


def main() -> None:
    k_grid = (4, 5, 6, 7)
    out: dict = {
        "method": "Compare closest-k local smile fit via median abs pricing residual ticks.",
        "k_grid": list(k_grid),
        "by_day": {},
    }
    for day in (0, 1, 2):
        df = pd.read_csv(REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv", sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        idx = pvt.index.to_numpy()
        rng = np.random.default_rng(101 + day)
        sample = sorted(rng.choice(idx, size=min(700, len(idx)), replace=False).tolist())
        errs = {k: [] for k in k_grid}

        for ts in sample:
            row = pvt.loc[ts]
            s = float(row["VELVETFRUIT_EXTRACT"])
            t = dte_eff(day, int(ts)) / 365.0
            pairs: list[tuple[float, float, float]] = []
            obs: dict[int, float] = {}
            for k in STRIKES:
                sym = f"VEV_{k}"
                if sym not in row or pd.isna(row[sym]):
                    continue
                px = float(row[sym])
                iv = iv_bisect(px, s, float(k), t)
                if iv is None:
                    continue
                pairs.append((float(k), abs(float(k) - s), iv))
                obs[k] = px
            if len(pairs) < 3:
                continue

            for kg in k_grid:
                coeffs = fit_surface(s, pairs, kg)
                if coeffs is None:
                    continue
                for k, px in obs.items():
                    sig = iv_surface_value(s, float(k), coeffs)
                    if sig is None:
                        continue
                    theo = bs_call(s, float(k), t, sig)
                    errs[kg].append(abs(theo - px))

        out["by_day"][str(day)] = {
            str(k): {
                "n": len(errs[k]),
                "median_abs_tick_resid": float(np.median(errs[k])) if errs[k] else None,
                "p90_abs_tick_resid": float(np.quantile(errs[k], 0.9)) if errs[k] else None,
            }
            for k in k_grid
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
