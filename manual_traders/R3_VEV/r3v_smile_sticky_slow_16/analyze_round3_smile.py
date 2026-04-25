"""
One-off analysis for r3v_smile_sticky_slow_16: IV smile + vega weights + spread stats on Round 3 tapes.

DTE / T: same conventions as round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py
and round3work/round3description.txt (CSV day 0→8d open, 1→7d, 2→6d; intraday winding).
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

# manual_traders/R3_VEV/r3v_smile_sticky_slow_16/ → repo root is four levels up
REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT_JSON = Path(__file__).resolve().parent / "analysis_outputs" / "smile_vega_spread_summary.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years_effective(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return S * norm.pdf(d1) * math.sqrt(T)


def implied_vol_call(market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-9:
        return float("nan")
    if market >= S - 1e-9:
        return float("nan")
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    lo, hi = 1e-5, 15.0
    try:
        fl, fh = f(lo), f(hi)
        if fl > 0 or fh < 0:
            return float("nan")
        return brentq(f, lo, hi, xtol=1e-8, rtol=1e-8)
    except ValueError:
        return float("nan")


def spread_width(row: pd.Series) -> float | None:
    bp = row.get("bid_price_1")
    ap = row.get("ask_price_1")
    if pd.isna(bp) or pd.isna(ap):
        return None
    return float(ap) - float(bp)


def main() -> None:
    step = 50
    rows_iv: list[dict] = []
    spread_by_v: dict[str, list[float]] = {v: [] for v in VOUCHERS}
    spread_ex: list[float] = []

    for csv_day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{csv_day}.csv"
        df = pd.read_csv(path, sep=";")
        ts_list = sorted(df["timestamp"].unique())[::step]
        for ts in ts_list:
            sub = df[df["timestamp"] == ts]
            ex = sub[sub["product"] == "VELVETFRUIT_EXTRACT"]
            if ex.empty:
                continue
            S = float(ex.iloc[0]["mid_price"])
            if S <= 0:
                continue
            T = t_years_effective(csv_day, int(ts))
            sw = spread_width(ex.iloc[0])
            if sw is not None:
                spread_ex.append(sw)

            for v in VOUCHERS:
                r0 = sub[sub["product"] == v]
                if r0.empty:
                    continue
                r = r0.iloc[0]
                mid = float(r["mid_price"])
                K = float(v.split("_")[1])
                iv = implied_vol_call(mid, S, K, T, 0.0)
                if not np.isfinite(iv):
                    continue
                veg = bs_vega(S, K, T, iv, 0.0)
                swv = spread_width(r)
                if swv is not None:
                    spread_by_v[v].append(swv)
                m_t = math.log(K / S) / math.sqrt(T) if T > 0 else float("nan")
                rows_iv.append(
                    {
                        "csv_day": csv_day,
                        "timestamp": int(ts),
                        "voucher": v,
                        "S": S,
                        "mid": mid,
                        "iv": iv,
                        "vega": veg,
                        "m_t": m_t,
                    }
                )

    if not rows_iv:
        print("no IV rows", file=sys.stderr)
        sys.exit(1)

    panel = pd.DataFrame(rows_iv)
    panel["abs_m"] = panel["m_t"].abs()

    # vega-weighted IV level (ATM-ish m_t in [-0.5, 0.5])
    atm = panel[panel["abs_m"] <= 0.5]
    w_iv = float((atm["iv"] * atm["vega"]).sum() / max(atm["vega"].sum(), 1e-9))

    # smile curvature proxy: IV at high m vs low m (by strike region)
    hi = panel[panel["m_t"] > 0.3]["iv"].median()
    lo = panel[panel["m_t"] < -0.3]["iv"].median()

    out = {
        "dte_mapping": "round3description.txt + plot_iv_smile_round3: csv_day 0→DTE8 at open, 1→7, 2→6; intraday DTE winds ~1 day per session",
        "subsample_timestamp_step": step,
        "n_iv_points": len(panel),
        "iv_median_all": float(panel["iv"].median()),
        "iv_iqr_all": float(panel["iv"].quantile(0.75) - panel["iv"].quantile(0.25)),
        "vega_weighted_iv_atm_band": w_iv,
        "iv_median_m_t_gt_0.3": float(hi) if np.isfinite(hi) else None,
        "iv_median_m_t_lt_-0.3": float(lo) if np.isfinite(lo) else None,
        "spread_median_VELVETFRUIT_EXTRACT": float(np.median(spread_ex)) if spread_ex else None,
        "spread_median_by_voucher": {
            v: float(np.median(spread_by_v[v])) if spread_by_v[v] else None for v in VOUCHERS
        },
        "interpretation": "Sticky slow fair: cross-sectional IV is fairly smooth; use EWMA on fitted smile coeffs + vega-weighted edge vs mids.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
