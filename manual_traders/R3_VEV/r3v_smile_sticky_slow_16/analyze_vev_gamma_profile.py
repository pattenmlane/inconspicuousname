"""
Tape: BS call gamma at market IV for each VEV strike; days 0-2, subsampled timestamps.

Uses same dte_effective / T as traders; S = extract mid; IV from mid via brentq.
Writes analysis_outputs/vev_gamma_by_strike_median.json
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
OUT = REPO / "manual_traders" / "R3_VEV" / "r3v_smile_sticky_slow_16" / "analysis_outputs" / "vev_gamma_by_strike_median.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
V = [f"VEV_{k}" for k in STRIKES]


def dte_e(day: int, ts: int) -> float:
    return max(8.0 - float(day) - (int(ts) // 100) / 10000.0, 1e-6)


def t_y(day: int, ts: int) -> float:
    return dte_e(day, ts) / 365.0


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


def call_gamma(S: float, K: float, T: float, s: float) -> float:
    if T <= 0 or s <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = s * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s * T) / v
    return float(norm.pdf(d1) / (S * v))


def main() -> None:
    step = 200
    by: dict[str, list[float]] = {vv: [] for vv in V}
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        for ts in sorted(df["timestamp"].unique())[::step]:
            sub = df[df["timestamp"] == ts]
            ex = sub[sub["product"] == "VELVETFRUIT_EXTRACT"]
            if ex.empty:
                continue
            S = float(ex.iloc[0]["mid_price"])
            Tq = t_y(day, int(ts))
            for vv in V:
                r = sub[sub["product"] == vv]
                if r.empty:
                    continue
                row = r.iloc[0]
                if pd.isna(row["bid_price_1"]) or pd.isna(row["ask_price_1"]):
                    continue
                mid = 0.5 * (float(row["bid_price_1"]) + float(row["ask_price_1"]))
                K = float(vv.split("_")[1])
                iv0 = ivv(mid, S, K, Tq)
                if iv0 is None:
                    continue
                g = call_gamma(S, K, Tq, iv0)
                if math.isfinite(g):
                    by[vv].append(g)

    med = {k: float(np.median(v)) if v else None for k, v in by.items()}
    # typical ATM ref ~ median of 5200-5300
    g_ref = 0.5 * (med.get("VEV_5200") or 0) + 0.5 * (med.get("VEV_5300") or 0)
    payload = {
        "dte": "8-csv_day at open, intraday wind per session",
        "subsample": step,
        "median_call_gamma_by_voucher": med,
        "suggested_GAMMA_REF_from_tape_median_5200_5300": g_ref,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
