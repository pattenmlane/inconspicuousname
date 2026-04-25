"""
Compare IV smile fit: quadratic vs cubic in m_t = log(K/S)/sqrt(T) on Round-3 tapes.
T: csv day 0/1/2 open DTE 8/7/6 + intraday wind-down; BS IV, r=0, time-value filter 4.5% of mid.
Outputs quad_vs_cubic_smile_v12.json with mean RMSE and mean |resid| on wing strikes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "quad_vs_cubic_smile_v12.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
MIN_TV = 0.045
WINGS = (4000, 4500, 5400, 5500, 6000, 6500)
STEP = 100


def t_years(day: int, ts: int) -> float:
    dte = max(8.0 - float(day) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte / 365.0


def bs(S: float, K: float, T: float, s: float) -> float:
    if T <= 0 or s <= 1e-12:
        return max(S - K, 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def iv_mid(m: float, S: float, K: float, T: float) -> float | None:
    if m <= max(S - K, 0) + 1e-6 or m >= S - 1e-6 or S <= 0:
        return None

    def f(x: float) -> float:
        return bs(S, K, T, x) - m

    if f(1e-4) > 0 or f(12.0) < 0:
        return None
    return float(brentq(f, 1e-4, 12.0))


def main() -> None:
    rmse2_all: list[float] = []
    rmse3_all: list[float] = []
    absr2: list[float] = []
    absr3: list[float] = []

    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        p = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        p = p[[U] + [c for c in VOU if c in p.columns]].sort_index()
        for ts in p.index[::STEP]:
            row = p.loc[ts]
            S = float(row[U])
            T = t_years(day, int(ts))
            if S <= 0 or T <= 0:
                continue
            srt = math.sqrt(T)
            xs, ys = [], []
            for v in VOU:
                if v not in row.index or pd.isna(row[v]):
                    continue
                m = float(row[v])
                K = int(v.split("_")[1])
                intr = max(S - K, 0.0)
                if m <= 0 or (m - intr) / m < MIN_TV:
                    continue
                sig = iv_mid(m, S, K, T)
                if sig is None or not np.isfinite(sig):
                    continue
                xs.append(math.log(K / S) / srt)
                ys.append(sig)
            if len(xs) < 7:
                continue
            xf, yf = np.asarray(xs), np.asarray(ys)
            c2 = np.polyfit(xf, yf, 2)
            c3 = np.polyfit(xf, yf, 3)
            p2 = np.poly1d(c2)
            p3 = np.poly1d(c3)
            r2 = yf - p2(xf)
            r3 = yf - p3(xf)
            rmse2_all.append(float(np.sqrt(np.mean(r2**2))))
            rmse3_all.append(float(np.sqrt(np.mean(r3**2))))
            for k in WINGS:
                v = f"VEV_{k}"
                if v not in row.index or pd.isna(row[v]):
                    continue
                m = float(row[v])
                sig = iv_mid(m, S, k, T)
                if sig is None:
                    continue
                m_t = math.log(k / S) / srt
                absr2.append(abs(float(sig - p2(m_t))))
                absr3.append(abs(float(sig - p3(m_t))))

    pay = {
        "method": "BS IV (r=0), m_t=log(K/S)/sqrt(T), polyfit deg 2 vs 3; subsample every 100 ticks; MIN_TV=0.045 of mid for smile points",
        "n_fit_points": "require >=7 IV points per timestamp for cubic",
        "mean_rmse_all_strikes_used_in_fit": {
            "quadratic": float(np.mean(rmse2_all)) if rmse2_all else None,
            "cubic": float(np.mean(rmse3_all)) if rmse3_all else None,
        },
        "mean_abs_residual_wing_strikes_only": {
            "quadratic": float(np.mean(absr2)) if absr2 else None,
            "cubic": float(np.mean(absr3)) if absr3 else None,
        },
        "n_quadratic_fits": len(rmse2_all),
    }
    OUT.write_text(json.dumps(pay, indent=2), encoding="utf-8")
    print(OUT, json.dumps(pay, indent=2))


if __name__ == "__main__":
    main()
