"""Tape slice: per-timestamp IV smile skew vs top-of-book spread on a focal VEV (5200)."""
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
OUT = Path(__file__).resolve().parent / "analysis_outputs"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]
FOCAL = "VEV_5200"
U = "VELVETFRUIT_EXTRACT"


def t_years(day: int, ts: int) -> float:
    dte = max(float(8 - int(day)) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte / 365.0


def bs(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def iv(mkt: float, S: float, K: float, T: float) -> float:
    intr = max(S - K, 0.0)
    if mkt <= intr + 1e-9 or mkt >= S - 1e-9 or S <= 0 or T <= 0:
        return float("nan")

    def f(s: float) -> float:
        return bs(S, K, T, s) - mkt

    try:
        if f(1e-5) > 0 or f(15.0) < 0:
            return float("nan")
        return float(brentq(f, 1e-5, 15.0))
    except ValueError:
        return float("nan")


def top_spread_from_df(df: pd.DataFrame, product: str, ts: int) -> float | None:
    r = df[(df["timestamp"] == ts) & (df["product"] == product)]
    if r.empty:
        return None
    row = r.iloc[0]
    bps, aps = [], []
    for i in (1, 2, 3):
        bp, ap = row.get(f"bid_price_{i}"), row.get(f"ask_price_{i}")
        bv, av = row.get(f"bid_volume_{i}"), row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and int(bv) > 0:
            bps.append(float(bp))
        if pd.notna(ap) and pd.notna(av) and int(av) > 0:
            aps.append(float(ap))
    if not bps or not aps:
        return None
    return min(aps) - max(bps)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        for ts in sorted(pvt.index)[::200]:
            if U not in pvt.columns:
                continue
            S = float(pvt.loc[ts, U])
            if S <= 0:
                continue
            T = t_years(day, int(ts))
            xs, ys = [], []
            for k, v in zip(STRIKES, VOU):
                if v not in pvt.columns:
                    continue
                mid = float(pvt.loc[ts, v])
                sig = iv(mid, S, float(k), T)
                if not np.isfinite(sig):
                    continue
                m_t = math.log(float(k) / S) / math.sqrt(T)
                xs.append(m_t)
                ys.append(sig)
            if len(xs) < 5:
                continue
            xf, yf = np.array(xs), np.array(ys)
            a, b = np.polyfit(xf, yf, 1)
            resid = yf - (a * xf + b)
            skew = float(resid[xf.argmax()] - resid[xf.argmin()])
            spr = top_spread_from_df(df, FOCAL, int(ts))
            rows.append(
                {
                    "day": day,
                    "timestamp": int(ts),
                    "slope_iv_vs_m": float(a),
                    "skew_resid_m_max_minus_m_min": skew,
                    "focal_top_spread": spr,
                }
            )
    out = pd.DataFrame(rows)
    p_csv = OUT / "iv_smile_skew_vs_spread_5200.csv"
    out.to_csv(p_csv, index=False)
    valid = out.dropna(subset=["focal_top_spread"])
    corr = float(valid["skew_resid_m_max_minus_m_min"].corr(valid["focal_top_spread"])) if len(valid) > 3 else None
    summ = {
        "focal": FOCAL,
        "n_rows": int(len(out)),
        "corr_skew_vs_focal_top_spread": corr,
        "method": "IV per strike from mid (BS, r=0), linear fit IV vs m_t; skew = resid at max m - resid at min m; m=log(K/S)/sqrt(T). Spread = min ask - max bid for focal row.",
    }
    (OUT / "iv_smile_skew_vs_spread_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("Wrote", p_csv, "corr", corr)


if __name__ == "__main__":
    main()
