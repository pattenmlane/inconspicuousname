#!/usr/bin/env python3
"""Per STRATEGY: inner-join 5200+5300+extract. For each row, invert IV from VEV mid (BS) at each
strike; report median implied vol when joint gate tight vs not (per strike / pooled ATM-ish).
Pooled days 0-2. Motivates BBO-IV fair in trader_v27.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "bbo_iv_tight_vs_wide_d0_2.json"
TH = 2
STRIKES = (5000, 5100, 5200, 5300, 5400)


def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(s: float, k: float, t: float, sig: float) -> float:
    if t <= 1e-12 or sig <= 1e-12:
        return max(s - k, 0.0)
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / v
    d2 = d1 - v
    return s * ncdf(d1) - k * ncdf(d2)


def iv(mid: float, s: float, k: float, t: float) -> float | None:
    intr = max(s - k, 0.0)
    if mid <= intr + 1e-6 or mid >= s - 1e-6 or s <= 0 or k <= 0 or t <= 1e-12:
        return None
    lo, hi = 1e-4, 12.0
    if bs(s, k, t, lo) - mid > 0 or bs(s, k, t, hi) - mid < 0:
        return None
    for _ in range(30):
        m = 0.5 * (lo + hi)
        if bs(s, k, t, m) >= mid:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def dte(d: int, ts: int) -> float:
    return max(8.0 - float(d) - ((int(ts) // 100) / 10_000.0), 1e-6)


def one(df: pd.DataFrame, p: str) -> pd.DataFrame:
    v = (
        df[df["product"] == p]
        .drop_duplicates("timestamp", keep="first")
        .sort_values("timestamp")
    )
    b = pd.to_numeric(v["bid_price_1"], errors="coerce")
    a = pd.to_numeric(v["ask_price_1"], errors="coerce")
    m = pd.to_numeric(v["mid_price"], errors="coerce")
    v = v.assign(spread=(a - b).astype(float), mid=m)
    return v[["timestamp", "spread", "mid"]].copy()


def run_day(day: int) -> dict:
    df = pd.read_csv(REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv", sep=";")
    a = one(df, "VEV_5200").rename(columns={"spread": "s5200"})
    b = one(df, "VEV_5300").rename(columns={"spread": "s5300"})
    e = one(df, "VELVETFRUIT_EXTRACT").rename(columns={"mid": "s"})
    m = a.merge(b, on="timestamp").merge(e[["timestamp", "s"]], on="timestamp")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    per_k: dict = {}
    for k in STRIKES:
        sym = f"VEV_{k}"
        o = one(df, sym)
        mid_col = f"mid_{k}"
        o2 = o.rename(columns={"mid": mid_col})
        mm = m.merge(o2, on="timestamp", how="inner").dropna()
        ivs_t: list[float] = []
        ivs_w: list[float] = []
        for row in mm.itertuples():
            t = dte(day, int(row.timestamp)) / 365.0
            sig = iv(float(getattr(row, mid_col)), float(row.s), float(k), t)
            if sig is None:
                continue
            if bool(row.tight):
                ivs_t.append(sig)
            else:
                ivs_w.append(sig)
        per_k[str(k)] = {
            "median_iv_tight": float(np.median(ivs_t)) if ivs_t else None,
            "median_iv_wide": float(np.median(ivs_w)) if ivs_w else None,
            "n_tight": len(ivs_t),
            "n_wide": len(ivs_w),
        }
    return per_k


def main() -> None:
    out = {
        "method": "Bisect Black-Scholes IV from VEV mid vs extract mid, per timestamp; TH=2 joint gate. Days 0-2 only.",
        "by_day": {str(d): run_day(d) for d in (0, 1, 2)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
