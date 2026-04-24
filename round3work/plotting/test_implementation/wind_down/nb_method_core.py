"""
Notebook `round3.ipynb` method (Prosperity 3 analysis) adapted to P4 Round 3 data.

- T (years) = (days_to_expiry_open - t / 10_000) / 365  with t = row index 0..N-1 in time order
  (**intraday / session wind-down**, same spirit as `plot_iv_smile_round3` for historical days).
- Moneyness for smile: m = log(K/S) / sqrt(T)
- IV: bisection in [0.01, 1.0] on vol (same as notebook implied_volatility)
- Smile: np.polyfit(m, iv, deg=2) each timestamp across all strikes
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

_OM = Path(__file__).resolve().parent.parent.parent / "original_method" / "wind_down" / "combined_analysis"
if str(_OM) not in sys.path:
    sys.path.insert(0, str(_OM))

from plot_iv_smile_round3 import (
    STRIKES,
    VOUCHERS,
    dte_from_csv_day,
    fit_smile_poly,
    load_day_wide,
    subsample_wide,
)

# .../test_implementation/wind_down/this_file → repo = five .parent hops.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "Prosperity4Data" / "ROUND_3"


def _cdf(x: float) -> float:
    return float(norm.cdf(x))


def black_scholes(
    asset_price: float,
    strike_price: float,
    expiration_time: float,
    risk_free_rate: float,
    volatility: float,
) -> float:
    if expiration_time <= 0 or volatility <= 1e-12:
        return max(asset_price - strike_price, 0.0)
    d1 = (
        math.log(asset_price / strike_price)
        + (risk_free_rate + volatility**2 / 2) * expiration_time
    ) / (volatility * math.sqrt(expiration_time))
    d2 = d1 - volatility * math.sqrt(expiration_time)
    return asset_price * _cdf(d1) - strike_price * math.exp(
        -risk_free_rate * expiration_time
    ) * _cdf(d2)


def implied_volatility_nb(
    asset_price: float,
    strike_price: float,
    voucher_price: float,
    expiration_time: float,
    risk_free_rate: float = 0.0,
    lo: float = 0.01,
    hi: float = 1.0,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> float:
    """Exact notebook bisection (round3.ipynb cell 3)."""
    if expiration_time <= 0 or asset_price <= 0 or strike_price <= 0:
        return float("nan")
    volatility = (lo + hi) / 2
    for _ in range(max_iterations):
        expected_price = black_scholes(
            asset_price, strike_price, expiration_time, risk_free_rate, volatility
        )
        delta = expected_price - voucher_price
        if abs(delta) < tolerance:
            break
        if delta > 0:
            hi = volatility
        else:
            lo = volatility
        volatility = (lo + hi) / 2
    return float(volatility)


def expiration_time_years(dte_open: int, t_index: int) -> float:
    """Notebook: days_to_expiry/365 - (t/10_000/365)."""
    return max((float(dte_open) - float(t_index) / 10_000.0) / 365.0, 1e-12)


def m_notebook(S: float, K: float, T_years: float) -> float:
    """m = log(K/S) / sqrt(T) — notebook smile x-axis."""
    if S <= 0 or K <= 0 or T_years <= 0:
        return float("nan")
    return math.log(K / S) / math.sqrt(T_years)


def index_map_timestamp_to_row_idx(wide: pd.DataFrame) -> dict[int, int]:
    wide = wide.sort_index()
    return {int(ts): i for i, ts in enumerate(wide.index)}


def compute_iv_panel_nb(wide: pd.DataFrame, t_index_map: dict[int, int]) -> pd.DataFrame:
    """IV panel using notebook T and m; wide may be subsampled but t_index_map is full-day indices."""
    rows = []
    day = int(wide["day"].iloc[0])
    d0 = dte_from_csv_day(day)
    for ts, row in wide.sort_index().iterrows():
        ts_i = int(ts)
        t_idx = t_index_map[ts_i]
        T = expiration_time_years(d0, t_idx)
        S = float(row["S"])
        for v in VOUCHERS:
            if v not in row.index:
                continue
            K = int(v.split("_")[1])
            mid = float(row[v])
            iv = implied_volatility_nb(S, K, mid, T, 0.0)
            mnb = m_notebook(S, K, T)
            rows.append(
                {
                    "timestamp": ts_i,
                    "day": day,
                    "dte_open": d0,
                    "t_row": t_idx,
                    "T_years": float(T),
                    "voucher": v,
                    "K": K,
                    "S": S,
                    "mid": mid,
                    "m_nb": float(mnb),
                    "iv": float(iv),
                }
            )
    return pd.DataFrame(rows)


def build_ivdf_nb_all_days(step: int = 20) -> pd.DataFrame:
    parts = []
    for day in (0, 1, 2):
        wf = load_day_wide(day).sort_index()
        mp = index_map_timestamp_to_row_idx(wf)
        wsub = subsample_wide(wf, step=step)
        parts.append(compute_iv_panel_nb(wsub, mp))
    return pd.concat(parts, ignore_index=True)


def fit_smile_residuals_nb(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return fit_smile_poly(g, "m_nb", "iv")


def build_resdf_nb_from_ivdf(ivdf: pd.DataFrame) -> pd.DataFrame:
    res_records = []
    for day in (0, 1, 2):
        sub = ivdf[ivdf["day"] == day]
        for ts in sub["timestamp"].unique():
            g = sub[sub["timestamp"] == ts].copy()
            fit, res, _ = fit_smile_residuals_nb(g)
            g = g.assign(iv_fit=fit, iv_res=res)
            res_records.append(g)
    return pd.concat(res_records, ignore_index=True)


def build_full_resdf_nb_day(day: int) -> pd.DataFrame:
    """Full session, all strikes — for Frankfurt-style time series and fig6a."""
    wf = load_day_wide(day).sort_index()
    mp = index_map_timestamp_to_row_idx(wf)
    rows: list[dict] = []
    d0 = dte_from_csv_day(day)
    for ts, row in wf.iterrows():
        ts_i = int(ts)
        t_idx = mp[ts_i]
        T = expiration_time_years(d0, t_idx)
        S = float(row["S"])
        glist = []
        for v in VOUCHERS:
            if v not in row.index:
                continue
            K = int(v.split("_")[1])
            mid = float(row[v])
            iv = implied_volatility_nb(S, K, mid, T, 0.0)
            mnb = m_notebook(S, K, T)
            glist.append(
                {
                    "timestamp": ts_i,
                    "day": day,
                    "dte_open": d0,
                    "t_row": t_idx,
                    "T_years": float(T),
                    "voucher": v,
                    "K": K,
                    "S": S,
                    "mid": mid,
                    "m_nb": float(mnb),
                    "iv": float(iv),
                }
            )
        g = pd.DataFrame(glist)
        if len(g) < 4:
            continue
        fit, res, _ = fit_smile_residuals_nb(g)
        for i in range(len(g)):
            ivf = float(fit[i])
            mid_i = float(g["mid"].iloc[i])
            Ki = float(g["K"].iloc[i])
            theo = (
                black_scholes(S, Ki, T, 0.0, ivf)
                if np.isfinite(ivf) and S > 0 and Ki > 0 and T > 0
                else float("nan")
            )
            rows.append(
                {
                    "timestamp": ts_i,
                    "day": int(g["day"].iloc[i]),
                    "dte_open": int(g["dte_open"].iloc[i]),
                    "t_row": int(g["t_row"].iloc[i]),
                    "T_years": float(T),
                    "voucher": str(g["voucher"].iloc[i]),
                    "K": int(g["K"].iloc[i]),
                    "S": S,
                    "mid": mid_i,
                    "m_nb": float(g["m_nb"].iloc[i]),
                    "iv": float(g["iv"].iloc[i]),
                    "iv_fit": ivf,
                    "iv_res": float(res[i]),
                    "theoretical_mid": theo,
                    "price_dev": mid_i - theo if np.isfinite(theo) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def load_book_product(day: int, product: str) -> pd.DataFrame:
    path = DATA_DIR / f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    return df[df["product"] == product].sort_values("timestamp").reset_index(drop=True)
