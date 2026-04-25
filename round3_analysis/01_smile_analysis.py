"""Reproduce the IV smile + per-strike residual analysis on Prosperity4 ROUND_3.

DTE convention (matches the existing round3work tooling):
  CSV day 0 -> 8 days at open
  CSV day 1 -> 7 days at open
  CSV day 2 -> 6 days at open
Production round 3 -> 5 days at open.

T_years = (DTE_at_open * STEPS_PER_DAY - timestamp_step_index) / (365 * STEPS_PER_DAY)
where there are 1_000_000 / 100 = 10_000 ticks per day.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

DATA = Path("Prosperity4Data/ROUND_3")
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
UND = "VELVETFRUIT_EXTRACT"
STEPS_PER_DAY = 10_000  # 1_000_000 / 100
DTE_AT_OPEN = {0: 8, 1: 7, 2: 6}


def t_years(day: int, ts: int) -> float:
    """T in years; intraday wind-down."""
    step = ts // 100
    rem = DTE_AT_OPEN[day] * STEPS_PER_DAY - step
    return rem / (365.0 * STEPS_PER_DAY)


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def implied_vol(price: float, S: float, K: float, T: float) -> float:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-8 or T <= 0 or price >= S:
        return float("nan")
    try:
        return brentq(lambda s: bs_call(S, K, T, s) - price, 1e-4, 5.0, xtol=1e-7)
    except Exception:
        return float("nan")


def load_day_wide(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    products = [UND] + VOUCHERS
    df = df[df["product"].isin(products)]
    pivot = df.pivot_table(index="timestamp", columns="product", values="mid_price")
    pivot = pivot.rename(columns={UND: "S"})
    return pivot.sort_index()


def main():
    rows = []
    per_strike_resid = {k: [] for k in STRIKES}
    daily_atm = {}

    for day in (0, 1, 2):
        wide = load_day_wide(day)
        # subsample for speed (every 20th tick = 500 timestamps/day)
        sub = wide.iloc[::20]
        xs, ys = [], []
        per_k = {k: [] for k in STRIKES}
        for ts, row in sub.iterrows():
            S = float(row.get("S", float("nan")))
            if not math.isfinite(S) or S <= 0:
                continue
            T = t_years(day, int(ts))
            if T <= 0:
                continue
            sqrtT = math.sqrt(T)
            for K in STRIKES:
                v = f"VEV_{K}"
                p = float(row.get(v, float("nan")))
                if not math.isfinite(p):
                    continue
                iv = implied_vol(p, S, K, T)
                if not math.isfinite(iv):
                    continue
                m = math.log(K / S) / sqrtT
                xs.append(m)
                ys.append(iv)
                per_k[K].append((m, iv))
        if len(xs) < 50:
            continue
        xa = np.asarray(xs)
        ya = np.asarray(ys)
        a, b, c = np.polyfit(xa, ya, 2)
        daily_atm[day] = c
        rows.append((day, a, b, c, len(xa)))
        # residuals per strike against this day's parabola
        for K, lst in per_k.items():
            for m, iv in lst:
                fit = a * m * m + b * m + c
                per_strike_resid[K].append(iv - fit)

    print("\n=== Daily quadratic IV(m) fits  IV ~ a*m^2 + b*m + c  ===")
    print(f"{'day':>4} {'a':>10} {'b':>10} {'c (ATM IV)':>12} {'n':>7}")
    for d, a, b, c, n in rows:
        print(f"{d:>4} {a:10.5f} {b:10.5f} {c:12.5f} {n:>7}")

    print("\n=== Per-strike residual (IV - smile-fit) over all days ===")
    print(f"{'K':>6} {'mean_resid':>12} {'std':>10} {'n':>6}")
    for K in STRIKES:
        r = np.asarray(per_strike_resid[K])
        if r.size:
            print(f"{K:>6} {r.mean():12.5f} {r.std():10.5f} {r.size:>6}")
        else:
            print(f"{K:>6} {'N/A':>12} {'N/A':>10} {0:>6}")

    out = {
        "atm_iv_per_day": {str(k): float(v) for k, v in daily_atm.items()},
        "per_strike_resid_mean": {
            str(k): float(np.mean(per_strike_resid[k])) if per_strike_resid[k] else None
            for k in STRIKES
        },
        "per_strike_resid_std": {
            str(k): float(np.std(per_strike_resid[k])) if per_strike_resid[k] else None
            for k in STRIKES
        },
    }
    Path("round3_analysis/smile_summary.json").write_text(json.dumps(out, indent=2))
    print("\nSaved round3_analysis/smile_summary.json")


if __name__ == "__main__":
    main()
