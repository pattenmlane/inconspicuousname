"""
Offline: sweep extract EWMA rate used as S in IV/m_t smile fit; measure stability of
quad coeff c2 (curvature) path across subsampled timestamps. T from round3 DTE + intraday
winding (same dte as plot_iv_smile / trader).

Writes: analysis_outputs/anchor_ewma_smile_stability.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "anchor_ewma_smile_stability.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
V = [f"VEV_{k}" for k in STRIKES]


def dte_e(day: int, ts: int) -> float:
    return max(8.0 - float(day) - (int(ts) // 100) / 10000.0, 1e-6)


def t_y(d: int, ts: int) -> float:
    return dte_e(d, ts) / 365.0


def bsc(S: float, K: float, T: float, s: float) -> float:
    if T <= 0 or s <= 1e-12:
        return max(S - K, 0.0)
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def ivv(mid: float, S: float, K: float, T: float) -> float | None:
    if S <= 0 or K <= 0 or T <= 0 or mid <= max(S - K, 0) + 1e-6 or mid >= S - 1e-6:
        return None

    def f(sig: float) -> float:
        return bsc(S, K, T, sig) - mid

    try:
        if f(1e-5) > 0 or f(12) < 0:
            return None
        return float(brentq(f, 1e-5, 12, xtol=1e-7, rtol=1e-7))
    except ValueError:
        return None


def one_day(std_alpha: float, day: int, step: int) -> float:
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    c2s: list[float] = []
    s_ema: float | None = None
    for ts in sorted(df["timestamp"].unique())[::step]:
        sub = df.loc[df["timestamp"] == ts]
        ex = sub.loc[sub["product"] == "VELVETFRUIT_EXTRACT"]
        if ex.empty:
            continue
        s_raw = float(ex.iloc[0]["mid_price"])
        s_ema = s_raw if s_ema is None else (1.0 - std_alpha) * s_ema + std_alpha * s_raw
        S = float(s_ema)
        Tq = t_y(day, int(ts))
        st = math.sqrt(Tq)
        xs: list[float] = []
        ys: list[float] = []
        ws: list[float] = []
        for vv in V:
            r = sub.loc[sub["product"] == vv]
            if r.empty:
                continue
            row = r.iloc[0]
            if pd.isna(row["bid_price_1"]) or pd.isna(row["ask_price_1"]):
                continue
            mid = 0.5 * (float(row["bid_price_1"]) + float(row["ask_price_1"]))
            K = float(vv.split("_")[1])
            i = ivv(mid, S, K, Tq)
            if i is None:
                continue
            mt = math.log(K / S) / st
            d1 = (math.log(S / K) + 0.5 * i * i * Tq) / (i * math.sqrt(Tq))
            vg = S * norm.pdf(d1) * math.sqrt(Tq)
            xs.append(mt)
            ys.append(i)
            ws.append(max(vg, 1e-6))
        if len(xs) < 6:
            continue
        c = np.polyfit(np.asarray(xs), np.asarray(ys), 2, w=np.asarray(ws))
        c2s.append(float(c[0]))
    if len(c2s) < 4:
        return 999.0
    return float(np.std(c2s))


def main() -> None:
    step = 100
    rows = []
    for a in (0.006, 0.012, 0.02, 0.03, 0.045):
        s0, s1, s2 = [one_day(a, d, step) for d in (0, 1, 2)]
        rows.append(
            {
                "alpha": a,
                "std_c2_day0": s0,
                "std_c2_day1": s1,
                "std_c2_day2": s2,
                "sum_std_c2": s0 + s1 + s2,
            }
        )
    rows.sort(key=lambda r: r["sum_std_c2"])
    payload = {
        "dte": "8-csv_day at open, intraday -1d/session; t_years = dte_eff/365",
        "subsample": step,
        "note": "Lower sum_std_c2 = less oscillation in smile curvature when S uses that EWMA rate",
        "grid": rows,
        "best_alpha_by_smoothness": rows[0]["alpha"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT, "best_alpha", rows[0]["alpha"])


if __name__ == "__main__":
    main()
