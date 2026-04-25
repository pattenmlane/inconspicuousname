#!/usr/bin/env python3
"""Mean |vega| and |gamma| by strike (BS at spline IV) on Round 3 tapes; 1/500 subsample.
TTE: 8 - day_idx days; S = VELVETFRUIT_EXTRACT best-bid/ask size-weighted mid. Same outlier
IV drop as traders when IV range > 0.35. Writes analysis_outputs/vega_gamma_by_strike.csv
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
R = 0.0
SQRT2PI = math.sqrt(2.0 * math.pi)

VEV_SYMS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
STRIKES = {s: int(s.split("_")[1]) for s in VEV_SYMS}
EX = "VELVETFRUIT_EXTRACT"
STRIDE = 500


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm_cdf(d1) - K * math.exp(-R * T) * norm_cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    return S * math.sqrt(T) * norm_pdf(d1)


def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    return norm_pdf(d1) / (S * v)


def iv(S: float, K: float, T: float, price: float) -> float | None:
    if price <= 0 or bs_call(S, K, T, 4.5) < price - 1e-9:
        return None
    lo, hi = 1e-5, 4.5
    if bs_call(S, K, T, lo) > price:
        return None
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def wall(bb: float, bv: float, ba: float, av: float) -> float:
    t = bv + av
    if t <= 0:
        return 0.5 * (bb + ba)
    return (bb * av + ba * bv) / t


def fit(logks, ivs, robust: float) -> CubicSpline | None:
    if len(logks) < 4:
        return None
    x = np.array(logks, float)
    y = np.array(ivs, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    if (y.max() - y.min()) > robust and len(y) >= 6:
        med = float(np.median(y))
        drop = int(np.argmax(np.abs(y - med)))
        x = np.delete(x, drop)
        y = np.delete(y, drop)
    if len(x) < 4:
        return None
    return CubicSpline(x, y, bc_type="natural")


def run_day(day_idx: int) -> list[tuple[str, float, float, float]]:
    T = (8 - day_idx) / 365.25
    path = DATA / f"prices_round_3_day_{day_idx}.csv"
    rows = list(csv.DictReader(path.open(), delimiter=";"))
    by_ts: dict[int, list] = defaultdict(list)
    for r in rows:
        by_ts[int(r["timestamp"])].append(r)

    acc: dict[str, list[tuple[float, float]]] = {s: [] for s in VEV_SYMS}
    for ts, chunk in by_ts.items():
        if ts % STRIDE:
            continue
        prod: dict[str, object] = {}
        for r in chunk:
            prod[r["product"]] = r
        er = prod.get(EX)
        if not er:
            continue
        e = er  # type: ignore
        try:
            S = wall(
                float(e["bid_price_1"]),
                float(e["bid_volume_1"]),
                float(e["ask_price_1"]),
                float(e["ask_volume_1"]),
            )
        except (KeyError, ValueError, TypeError):
            continue
        lks, ivl = [], []
        for s in VEV_SYMS:
            x = prod.get(s)
            if not x:
                continue
            x = x  # type: ignore
            try:
                mid = wall(
                    float(x["bid_price_1"]),
                    float(x["bid_volume_1"]),
                    float(x["ask_price_1"]),
                    float(x["ask_volume_1"]),
                )
            except (KeyError, ValueError, TypeError):
                continue
            v = iv(S, STRIKES[s], T, mid)
            if v is None:
                continue
            lks.append(math.log(STRIKES[s]))
            ivl.append(v)
        cs = fit(lks, ivl, 0.35)
        if cs is None:
            continue
        for s in VEV_SYMS:
            K = float(STRIKES[s])
            try:
                sig = float(cs(math.log(K)))
            except Exception:
                continue
            vg = abs(bs_vega(S, K, T, max(sig, 1e-6)))
            g = abs(bs_gamma(S, K, T, max(sig, 1e-6)))
            acc[s].append((vg, g))

    out = []
    for s in VEV_SYMS:
        if not acc[s]:
            continue
        mvg = float(np.mean([a[0] for a in acc[s]]))
        mgg = float(np.mean([a[1] for a in acc[s]]))
        out.append((s, mvg, mgg, float(len(acc[s]))))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for d in (0, 1, 2):
        for s, mvg, mgg, n in run_day(d):
            rows.append({"day": d, "symbol": s, "mean_abs_vega": mvg, "mean_abs_gamma": mgg, "n": int(n)})
    p = OUT / "vega_gamma_by_strike.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["day", "symbol", "mean_abs_vega", "mean_abs_gamma", "n"]
        )
        w.writeheader()
        w.writerows(rows)
    print("wrote", p)


if __name__ == "__main__":
    main()
