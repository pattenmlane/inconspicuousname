#!/usr/bin/env python3
"""Greek diagnostics for local-smile concept on ROUND_3 tapes.

Method:
- Parse prices_round_3_day_{0,1,2}.csv (semicolon-separated).
- Use BS call model with r=0 and time convention aligned to round3description / plot_iv_smile_round3:
  CSV day d => DTE_open = 8-d, dte_eff = DTE_open - (timestamp//100)/10000, T=dte_eff/365.
- Compute implied vol from voucher mid.
- Fit local quadratic smile on closest-k strikes (k=6 default, compare k=3).
- Price each strike from fitted surface and compute mispricing in ticks.
- Compute vega and delta under fitted local surface.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "local_smile_greeks.json"

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(s: float, k: float, t: float, sigma: float) -> float:
    if t <= 1e-12 or sigma <= 1e-12:
        return max(s - k, 0.0)
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / v
    d2 = d1 - v
    return s * _norm_cdf(d1) - k * _norm_cdf(d2)


def bs_delta(s: float, k: float, t: float, sigma: float) -> float:
    if t <= 1e-12 or sigma <= 1e-12:
        return 1.0 if s > k else 0.0
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / v
    return _norm_cdf(d1)


def bs_vega(s: float, k: float, t: float, sigma: float) -> float:
    if t <= 1e-12 or sigma <= 1e-12:
        return 0.0
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / v
    return s * _norm_pdf(d1) * math.sqrt(t)


def implied_vol_bisect(price: float, s: float, k: float, t: float) -> float | None:
    intrinsic = max(s - k, 0.0)
    if price <= intrinsic + 1e-6 or price >= s - 1e-6:
        return None
    lo, hi = 1e-4, 12.0
    flo = bs_call_price(s, k, t, lo) - price
    fhi = bs_call_price(s, k, t, hi) - price
    if flo > 0 or fhi < 0:
        return None
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if bs_call_price(s, k, t, mid) >= price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def dte_eff(day: int, timestamp: int) -> float:
    d_open = 8.0 - float(day)
    intra = (int(timestamp) // 100) / 10_000.0
    return max(d_open - intra, 1e-6)


def fit_local_smile(s: float, iv_by_k: dict[int, float], closest_k: int) -> tuple[float, float, float] | None:
    valid = [(k, iv) for k, iv in iv_by_k.items() if iv == iv and iv > 0]
    if len(valid) < 2:
        return None
    valid.sort(key=lambda kv: abs(kv[0] - s))
    use = valid[: min(closest_k, len(valid))]
    xs = np.array([math.log(k / s) for k, _ in use], dtype=float)
    ys = np.array([iv for _, iv in use], dtype=float)
    deg = 2 if len(use) >= 3 else 1
    coeff = np.polyfit(xs, ys, deg=deg)
    if deg == 1:
        b, c = coeff
        return 0.0, float(b), float(c)
    a, b, c = coeff
    return float(a), float(b), float(c)


def iv_hat(s: float, k: float, coeff: tuple[float, float, float] | None) -> float | None:
    if coeff is None:
        return None
    a, b, c = coeff
    x = math.log(k / s)
    sig = a * x * x + b * x + c
    if sig <= 1e-5 or sig > 10:
        return None
    return sig


def load_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
    cols = ["VELVETFRUIT_EXTRACT"] + [v for v in VOUCHERS if v in pvt.columns]
    return pvt[cols].copy()


def analyze_day(day: int, sample_n: int = 800) -> dict:
    wide = load_day(day)
    idx = wide.index.to_numpy()
    rng = np.random.default_rng(100 + day)
    sample = sorted(rng.choice(idx, size=min(sample_n, len(idx)), replace=False).tolist())

    mis_k6 = []
    mis_k3 = []
    vega_abs = {k: [] for k in STRIKES}
    delta_abs = {k: [] for k in STRIKES}

    for ts in sample:
        row = wide.loc[ts]
        s = float(row["VELVETFRUIT_EXTRACT"])
        t = dte_eff(day, int(ts)) / 365.0

        iv_by_k: dict[int, float] = {}
        px_by_k: dict[int, float] = {}
        for k in STRIKES:
            sym = f"VEV_{k}"
            if sym not in row or pd.isna(row[sym]):
                continue
            px = float(row[sym])
            iv = implied_vol_bisect(px, s, float(k), t)
            if iv is None:
                continue
            iv_by_k[k] = iv
            px_by_k[k] = px
        if len(iv_by_k) < 4:
            continue

        coeff6 = fit_local_smile(s, iv_by_k, 6)
        coeff3 = fit_local_smile(s, iv_by_k, 3)

        for k, mkt in px_by_k.items():
            sig6 = iv_hat(s, float(k), coeff6)
            if sig6 is not None:
                theo6 = bs_call_price(s, float(k), t, sig6)
                mis_k6.append(theo6 - mkt)
                vega_abs[k].append(abs(bs_vega(s, float(k), t, sig6)))
                delta_abs[k].append(abs(bs_delta(s, float(k), t, sig6)))
            sig3 = iv_hat(s, float(k), coeff3)
            if sig3 is not None:
                theo3 = bs_call_price(s, float(k), t, sig3)
                mis_k3.append(theo3 - mkt)

    def q(a: list[float], p: float) -> float:
        return float(np.quantile(np.array(a, dtype=float), p)) if a else 0.0

    return {
        "n_samples_used": len(sample),
        "mispricing_ticks_k6": {
            "median_abs": q([abs(x) for x in mis_k6], 0.5),
            "p90_abs": q([abs(x) for x in mis_k6], 0.9),
        },
        "mispricing_ticks_k3": {
            "median_abs": q([abs(x) for x in mis_k3], 0.5),
            "p90_abs": q([abs(x) for x in mis_k3], 0.9),
        },
        "median_abs_vega_by_strike": {str(k): q(v, 0.5) for k, v in vega_abs.items()},
        "median_abs_delta_by_strike": {str(k): q(v, 0.5) for k, v in delta_abs.items()},
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "BS implied vol/greeks from ROUND_3 mids, local smile fit closest-k (k=6 vs k=3).",
        "day_summaries": {str(d): analyze_day(d) for d in (0, 1, 2)},
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
