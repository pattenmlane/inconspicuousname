"""
Vega-weighted mispricing (mid - BS_theo) vs mean top-of-book spread across VEVs.

Theo: BS call with sigma = model_iv from global smile (same coeffs as trader_v0).
Vega: at model_iv per strike. Spread: min(ask)-max(bid) from price CSV top level.
Timing: T from round3 historical day + intraday wind-down (DTE_eff/365).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
U = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]
_COEFFS = (0.14215151147708086, -0.0016298611395181932, 0.23576325646627055)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * _cdf(d1) - K * _cdf(d2)


def vega(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    return S * _pdf(d1) * math.sqrt(T)


def model_iv(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return 0.25
    m_t = math.log(K / S) / math.sqrt(T)
    a, b, c = _COEFFS
    return max(((a * m_t) + b) * m_t + c, 1e-4)


def t_years(day: int, ts: int) -> float:
    dte = max(float(8 - int(day)) - (int(ts) // 100) / 10_000.0, 1e-6)
    return dte / 365.0


def top_spread(df: pd.DataFrame, product: str, ts: int) -> float | None:
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
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        for ts in sorted(pvt.index)[::200]:
            if U not in pvt.columns:
                continue
            S = float(pvt.loc[ts, U])
            if S <= 0:
                continue
            T = t_years(day, int(ts))
            num = 0.0
            den = 0.0
            spreads: list[float] = []
            for v, k in zip(VOU, STRIKES):
                if v not in pvt.columns:
                    continue
                mid = float(pvt.loc[ts, v])
                sig = model_iv(S, float(k), T)
                th = bs(S, float(k), T, sig)
                vg = vega(S, float(k), T, sig)
                num += vg * (mid - th)
                den += vg
                s = top_spread(df, v, int(ts))
                if s is not None:
                    spreads.append(s)
            if den <= 0 or not spreads:
                continue
            rows.append(
                {
                    "day": day,
                    "timestamp": int(ts),
                    "vega_w_mispr": float(num / den),
                    "mean_spread_vev": float(np.mean(spreads)),
                }
            )
    tbl = pd.DataFrame(rows)
    p_csv = OUT / "vega_w_mispricing_vs_mean_spread.csv"
    tbl.to_csv(p_csv, index=False)
    z = tbl.dropna()
    c = float(z["vega_w_mispr"].corr(z["mean_spread_vev"])) if len(z) > 3 else None
    summ = {
        "n_rows": int(len(tbl)),
        "corr_mispr_vs_spread": c,
        "method": "vega = S*φ(d1)*√T at model IV; theo = BS; mispricing = mid-theo. Mean spread = average over strikes with a valid book.",
    }
    (OUT / "vega_w_mispricing_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(summ)


if __name__ == "__main__":
    main()
