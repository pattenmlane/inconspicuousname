#!/usr/bin/env python3
"""Compare BS fair from in-sample spline IV vs leave-one-out IV at each strike (subsamp 1/500).
Outputs analysis_outputs/loo_vs_insample_fair_gap.csv (mean abs gap by strike/day)."""
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
VEV = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]
KMAP = {s: int(s.split("_")[1]) for s in VEV}
EX = "VELVETFRUIT_EXTRACT"
STRIDE = 500
ROBUST = 0.35


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


def wall(bb, bv, ba, av):
    t = float(bv) + float(av)
    if t <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * float(av) + float(ba) * float(bv)) / t


def nodes(logks, ivs):
    if len(logks) < 4:
        return None
    x = np.array(logks, float)
    y = np.array(ivs, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    if (y.max() - y.min()) > ROBUST and len(y) >= 6:
        med = float(np.median(y))
        drop = int(np.argmax(np.abs(y - med)))
        x, y = np.delete(x, drop), np.delete(y, drop)
    if len(x) < 4:
        return None
    return x, y


def idx(x, logk: float) -> int | None:
    for j in range(len(x)):
        if abs(float(x[j]) - logk) < 1e-5:
            return j
    return None


def loo_iv(x, y, logk: float) -> float:
    j = idx(x, logk)
    if j is None or len(x) < 4:
        cs = CubicSpline(x, y, bc_type="natural")
        return float(cs(float(logk)))
    if len(x) < 5:
        cs = CubicSpline(x, y, bc_type="natural")
        return float(cs(float(logk)))
    xl, yl = np.delete(x, j), np.delete(y, j)
    cs = CubicSpline(xl, yl, bc_type="natural")
    return float(cs(float(logk)))


def insample_iv(x, y, logk: float) -> float:
    return float(CubicSpline(x, y, bc_type="natural")(float(logk)))


def run_day(d: int) -> list[dict]:
    T = (8 - d) / 365.25
    rows = list(csv.DictReader((DATA / f"prices_round_3_day_{d}.csv").open(), delimiter=";"))
    by = defaultdict(list)
    for r in rows:
        by[int(r["timestamp"])].append(r)
    acc = defaultdict(list)
    for ts, chunk in by.items():
        if ts % STRIDE:
            continue
        pr = {r["product"]: r for r in chunk}
        e = pr.get(EX)
        if not e:
            continue
        try:
            S = wall(
                e["bid_price_1"],
                e["bid_volume_1"],
                e["ask_price_1"],
                e["ask_volume_1"],
            )
        except (KeyError, TypeError, ValueError):
            continue
        lks, vs = [], []
        for s in VEV:
            x = pr.get(s)
            if not x:
                continue
            try:
                mid = wall(
                    x["bid_price_1"],
                    x["bid_volume_1"],
                    x["ask_price_1"],
                    x["ask_volume_1"],
                )
            except (KeyError, TypeError, ValueError):
                continue
            v = iv(S, KMAP[s], T, float(mid))
            if v is None:
                continue
            lks.append(math.log(KMAP[s]))
            vs.append(v)
        xy = nodes(lks, vs)
        if xy is None:
            continue
        x, y = xy
        for s in VEV:
            if s not in pr:
                continue
            lk = math.log(KMAP[s])
            if idx(x, lk) is None and len(x) < len(VEV):
                continue
            try:
                sig_i = insample_iv(x, y, lk)
                sig_l = loo_iv(x, y, lk)
            except Exception:
                continue
            Kf = float(KMAP[s])
            f_i = bs_call(S, Kf, T, max(sig_i, 1e-6))
            f_l = bs_call(S, Kf, T, max(sig_l, 1e-6))
            acc[(d, s)].append(abs(f_l - f_i))
    out = []
    for (dd, s), g in acc.items():
        out.append(
            {
                "day": dd,
                "symbol": s,
                "mean_abs_fair_gap": float(np.mean(g)) if g else 0.0,
                "n": len(g),
            }
        )
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allr = []
    for d in (0, 1, 2):
        allr.extend(run_day(d))
    p = OUT / "loo_vs_insample_fair_gap.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["day", "symbol", "mean_abs_fair_gap", "n"])
        w.writeheader()
        w.writerows(allr)
    print("wrote", p, "rows", len(allr))


if __name__ == "__main__":
    main()
