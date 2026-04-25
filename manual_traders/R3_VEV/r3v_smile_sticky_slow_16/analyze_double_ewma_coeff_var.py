"""
Compare variance of smile quadratic coeff c2: single EWMA (0.045) vs two-layer
(fast 0.045 on raw fit, then slow 0.2 on the first EMA). Tape days 0-2, step 100.
Output: analysis_outputs/double_ewma_coeff_var.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "double_ewma_coeff_var.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
V = [f"VEV_{k}" for k in STRIKES]

FAST = 0.045
SLOW = 0.12  # match trader_v10 EWMA2_ALPHA


def T(day: int, ts: int) -> float:
    dte = max(8.0 - float(day) - (int(ts) // 100) / 10000.0, 1e-6)
    return dte / 365.0


def bsc(S: float, K: float, T: float, s: float) -> float:
    if T <= 0 or s <= 1e-12:
        return max(S - K, 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def ivv(mid: float, S: float, K: float, T: float) -> float | None:
    if mid <= max(S - K, 0) + 1e-6 or mid >= S - 1e-6 or S <= 0 or K <= 0 or T <= 0:
        return None

    def f(sig: float) -> float:
        return bsc(S, K, T, sig) - mid

    try:
        if f(1e-5) > 0 or f(12) < 0:
            return None
        return float(brentq(f, 1e-5, 12, xtol=1e-7, rtol=1e-7))
    except ValueError:
        return None


def main() -> None:
    step = 100
    ema = [0.15, 0.0, 0.24]
    ema_slow = [0.15, 0.0, 0.24]
    c2_one: list[float] = []
    c2_two: list[float] = []
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        for ts in sorted(df["timestamp"].unique())[::step]:
            sub = df.loc[df["timestamp"] == ts]
            ex = sub.loc[sub["product"] == "VELVETFRUIT_EXTRACT"]
            if ex.empty:
                continue
            S = float(ex.iloc[0]["mid_price"])
            Ty = T(day, int(ts))
            st = math.sqrt(Ty)
            xs, ys, ws = [], [], []
            for vv in V:
                r = sub.loc[sub["product"] == vv]
                if r.empty:
                    continue
                row = r.iloc[0]
                if pd.isna(row["bid_price_1"]) or pd.isna(row["ask_price_1"]):
                    continue
                mid = 0.5 * (float(row["bid_price_1"]) + float(row["ask_price_1"]))
                K = float(vv.split("_")[1])
                i = ivv(mid, S, K, Ty)
                if i is None:
                    continue
                mt = math.log(K / S) / st
                d1 = (math.log(S / K) + 0.5 * i * i * Ty) / (i * math.sqrt(Ty))
                vg = S * norm.pdf(d1) * math.sqrt(Ty)
                xs.append(mt)
                ys.append(i)
                ws.append(max(vg, 1e-6))
            if len(xs) < 6:
                continue
            c = list(np.polyfit(np.asarray(xs), np.asarray(ys), 2, w=np.asarray(ws)))
            a = FAST
            ema = [(1 - a) * ema[i] + a * c[i] for i in range(3)]
            c2_one.append(ema[0])
            b = SLOW
            ema_slow = [(1 - b) * ema_slow[i] + b * ema[i] for i in range(3)]
            c2_two.append(ema_slow[0])

    v1 = float(np.var(c2_one)) if c2_one else 0.0
    v2 = float(np.var(c2_two)) if c2_two else 0.0
    payload = {
        "step": step,
        "fast_alpha": FAST,
        "slow_alpha": SLOW,
        "n_steps": len(c2_one),
        "var_c2_single_layer_ewma": v1,
        "var_c2_double_layer_ewma": v2,
        "var_ratio_double_over_single": float(v2 / max(v1, 1e-18)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT, payload)


if __name__ == "__main__":
    main()
