#!/usr/bin/env python3
"""Median |BS call theta| by strike from tape IVs (Round 3). Theta proxy for time decay risk."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "theta_median_by_strike.json"

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call(s: float, k: float, t: float, sig: float) -> float:
    if t <= 1e-12 or sig <= 1e-12:
        return max(s - k, 0.0)
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / v
    d2 = d1 - v
    return s * norm_cdf(d1) - k * norm_cdf(d2)


def bs_theta_call(s: float, k: float, t: float, sig: float, r: float = 0.0) -> float:
    """Annualized BS call theta (classic sign: often negative for long options)."""
    if t <= 1e-12 or sig <= 1e-12:
        return 0.0
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / v
    d2 = d1 - v
    term1 = -s * norm_pdf(d1) * sig / (2.0 * math.sqrt(t))
    term2 = -r * k * math.exp(-r * t) * norm_cdf(d2)
    return term1 + term2


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
    out: dict = {
        "method": "Per row: IV from VEV mid via bisection, S=extract mid, T=dte_eff/365, r=0. "
        "Theta = BS call theta (annualized). Report median abs(theta) by strike over full tape.",
    }
    for day in (0, 1, 2):
        df = pd.read_csv(
            REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv",
            sep=";",
        )
        pvt = df.pivot_table(
            index="timestamp", columns="product", values="mid_price", aggfunc="first"
        )
        by_k: dict[int, list[float]] = {k: [] for k in STRIKES}
        for ts in pvt.index:
            if "VELVETFRUIT_EXTRACT" not in pvt.columns:
                continue
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
                th = bs_theta_call(s, float(k), t, iv, 0.0)
                by_k[k].append(abs(th))
        out[str(day)] = {
            str(k): {
                "n": len(by_k[k]),
                "median_abs_theta": float(np.median(by_k[k])) if by_k[k] else None,
            }
            for k in STRIKES
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
