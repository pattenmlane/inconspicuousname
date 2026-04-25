#!/usr/bin/env python3
"""
Tape study: for core strikes 5000–5500, compare IV to WLS smooth σ̂ from all 10 strike Newton-IVs
( wing weight 0.58, spline S=0.08, same T as v14 ). Averages mean |iv−σ̂| by strike over sampled rows.

Timing: csv day d -> DTE open = 8−d; dte_eff = dte_open - (ts//100)/10000; T = dte_eff/365; r=0.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent
OUT_FILE = OUT / "v14_wing_misfit_by_strike.json"

WING = {4000, 4500, 6000, 6500}
CORE = {5000, 5100, 5200, 5300, 5400, 5500}
W_WING, S_SMOOTH = 0.58, 0.08
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"


def t_years(day: int, ts: int) -> float:
    dte = max(8 - int(day) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte / 365.0


def bs(S: float, K: float, T: float, s: float) -> float:
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def vega(S: float, K: float, T: float, s: float) -> float:
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    return S * norm.pdf(d1) * math.sqrt(T)


def iv_n(mid: float, S: float, K: float, T: float, g: float) -> float:
    if T <= 0 or mid <= max(S - K, 0) + 1e-6 or mid >= S - 1e-9:
        return float("nan")
    s = max(min(g, 6.0), 0.04)
    for _ in range(10):
        e = bs(S, K, T, s) - mid
        if abs(e) < 1e-4:
            return s
        vg = vega(S, K, T, s)
        if vg < 1e-8:
            return float("nan")
        s -= e / vg
        s = max(min(s, 8.0), 0.03)
    return s if abs(bs(S, K, T, s) - mid) < 0.05 else float("nan")


def bba(row: dict) -> tuple[int | None, int | None]:
    bids, asks = [], []
    for i in (1, 2, 3):
        b, bv = row.get(f"bid_price_{i}"), row.get(f"bid_volume_{i}")
        a, av = row.get(f"ask_price_{i}"), row.get(f"ask_volume_{i}")
        if b and bv and int(float(bv)) > 0:
            bids.append(int(float(b)))
        if a and av and int(float(av)) > 0:
            asks.append(int(float(a)))
    if not bids or not asks:
        return None, None
    return max(bids), min(asks)


def fit_spline(xs: np.ndarray, vs: np.ndarray, ws: np.ndarray) -> UnivariateSpline | None:
    if len(xs) < 4:
        return None
    o = np.argsort(xs)
    try:
        return UnivariateSpline(xs[o], vs[o], w=ws[o], k=3, s=S_SMOOTH)
    except Exception:
        return None


def main() -> None:
    sums: dict[int, list[float]] = {k: [0.0, 0.0] for k in sorted(CORE)}  # abs_err, count
    n_rows = 0
    for day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{day}.csv"
        by_ts: dict[int, dict[str, dict]] = defaultdict(dict)
        with path.open() as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                p = row["product"]
                if p == U or p in VOUS:
                    by_ts[int(row["timestamp"])][p] = row
        for ts in sorted(by_ts.keys())[::200]:
            snap = by_ts[ts]
            if U not in snap:
                continue
            bbu, bau = bba(snap[U])
            if bbu is None:
                continue
            S = 0.5 * (bbu + bau)
            T = t_years(day, ts)
            xs, ivs, ws = [], [], []
            for v in VOUS:
                if v not in snap:
                    continue
                bb, ba = bba(snap[v])
                if bb is None or ba <= bb:
                    continue
                mid = 0.5 * (bb + ba)
                k = int(v.split("_")[1])
                iv = iv_n(mid, S, k, T, 0.45)
                if not math.isfinite(iv):
                    continue
                x = math.log(max(k / S, 1e-6))
                xs.append(x)
                ivs.append(iv)
                ws.append(W_WING if k in WING else 1.0)
            if len(xs) < 4:
                continue
            spl = fit_spline(np.array(xs), np.array(ivs), np.array(ws))
            if spl is None:
                continue
            for k in CORE:
                v = f"VEV_{k}"
                if v not in snap:
                    continue
                bb, ba = bba(snap[v])
                if bb is None or ba <= bb:
                    continue
                mid = 0.5 * (bb + ba)
                ivk = iv_n(mid, S, k, T, 0.45)
                if not math.isfinite(ivk):
                    continue
                xk = math.log(max(k / S, 1e-6))
                sig = float(spl(xk))
                sig = max(min(sig, 7.5), 0.03)
                sums[k][0] += abs(ivk - sig)
                sums[k][1] += 1.0
            n_rows += 1

    by_strike = {}
    for k, (a, c) in sums.items():
        by_strike[str(k)] = {
            "mean_abs_iv_minus_smooth": float(a / c) if c else None,
            "samples": int(c),
        }
    payload = {
        "WING_SPLINE_WEIGHT": W_WING,
        "SPLINE_SMOOTH": S_SMOOTH,
        "sampled_snapshot_rows": n_rows,
        "core_strikes_5000_5500": by_strike,
    }
    OUT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUT_FILE)


if __name__ == "__main__":
    main()
