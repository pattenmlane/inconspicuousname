#!/usr/bin/env python3
"""One-off tape analysis for r3v_realized_vol_regime_13 (Round 3 CSVs only)."""
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
OUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"

# Same as round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py
COEFFS = np.array([0.14215151147708086, -0.0016298611395181932, 0.23576325646627055])


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def implied_vol_call(market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-9 or market >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    lo, hi = 1e-5, 15.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return float("nan")
        return float(brentq(f, lo, hi, xtol=1e-8, rtol=1e-8))
    except ValueError:
        return float("nan")


def model_iv(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    m_t = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(COEFFS, m_t))


def nearest_strike(S: float) -> int:
    return min(STRIKES, key=lambda k: abs(float(k) - S))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    for day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        if U not in pvt.columns:
            raise RuntimeError("missing extract")
        ts_idx = sorted(pvt.index)[::200]
        prev_s: float | None = None
        log_rets: list[float] = []
        for ts in ts_idx:
            S = float(pvt.loc[ts, U])
            if S <= 0:
                continue
            T = t_years(day, int(ts))
            K0 = nearest_strike(S)
            v0 = f"VEV_{K0}"
            if v0 not in pvt.columns:
                continue
            mid0 = float(pvt.loc[ts, v0])
            implied_iv = implied_vol_call(mid0, S, float(K0), T, 0.0)
            iv_mod = model_iv(S, float(K0), T)
            if prev_s is not None and prev_s > 0:
                log_rets.append(math.log(S / prev_s))
            prev_s = S
            rv_short = float("nan")
            if len(log_rets) >= 5:
                sig = float(np.std(log_rets[-20:], ddof=1)) if len(log_rets) >= 20 else float(np.std(log_rets, ddof=1))
                rv_short = sig * math.sqrt(252.0 * 10_000.0)
            sprs: list[float] = []
            for v in VOUCHERS:
                if v not in pvt.columns:
                    continue
                sub = df[(df["timestamp"] == ts) & (df["product"] == v)]
                if sub.empty:
                    continue
                r0 = sub.iloc[0]
                bps = [r0.get(f"bid_price_{i}") for i in (1, 2, 3)]
                aps = [r0.get(f"ask_price_{i}") for i in (1, 2, 3)]
                bps = [float(x) for x in bps if pd.notna(x)]
                aps = [float(x) for x in aps if pd.notna(x)]
                if bps and aps:
                    sprs.append(min(aps) - max(bps))
            mean_spread = float(np.mean(sprs)) if sprs else float("nan")
            rows_out.append(
                {
                    "day": day,
                    "dte_open_calendar": dte_from_csv_day(day),
                    "timestamp": int(ts),
                    "T_years": T,
                    "S": S,
                    "atm_strike": K0,
                    "iv_market_atm": implied_iv,
                    "iv_model_smile_atm": iv_mod,
                    "iv_minus_model": float(implied_iv - iv_mod) if np.isfinite(implied_iv) and np.isfinite(iv_mod) else float("nan"),
                    "rv_ann_short_horizon": rv_short,
                    "iv_minus_rv": float(implied_iv - rv_short)
                    if np.isfinite(implied_iv) and np.isfinite(rv_short)
                    else float("nan"),
                    "mean_vev_spread": mean_spread,
                }
            )

    tbl = pd.DataFrame(rows_out)
    tbl_path = OUT_DIR / "rv_iv_atm_subsample200.csv"
    tbl.to_csv(tbl_path, index=False)

    summ = {
        "timing_assumptions": (
            "Calendar DTE at open of historical CSV day d: TTE = 8 - d (day 0->8d, 1->7d, 2->6d) per round3work/round3description.txt. "
            "Intraday: DTE_eff = start-of-day DTE minus (timestamp//100)/10000 (one full day over the session), "
            "T_years = DTE_eff/365, r=0, European call on VELVETFRUIT_EXTRACT; same as plot_iv_smile_round3.py."
        ),
        "iv_method": (
            "Black–Scholes call implied vol from voucher mid, S=extract mid, K=strike, T=T_years (brentq on sigma). "
            "Model IV uses global quadratic smile in m_t=log(K/S)/sqrt(T) with coeffs from fitted_smile_coeffs.json (5200_work)."
        ),
        "rv_method": (
            "Short-horizon realized vol of extract: rolling std of log returns over last min(20, len) subsample steps "
            "(200-timestamp spacing), annualized as sigma * sqrt(252 * 10000) treating each 100-timestamp step as 1/10000 day fraction "
            "at 10k steps/day (Frankfurt-style session length)."
        ),
        "n_rows": int(len(tbl)),
        "corr_iv_minus_rv_vs_mean_spread": float(tbl["iv_minus_rv"].corr(tbl["mean_vev_spread"]))
        if len(tbl) > 5
        else None,
        "mean_iv_minus_rv": float(tbl["iv_minus_rv"].mean(skipna=True)),
        "mean_iv_minus_model": float(tbl["iv_minus_model"].mean(skipna=True)),
    }
    summ_path = OUT_DIR / "rv_iv_regime_summary.json"
    summ_path.write_text(json.dumps(summ, indent=2), encoding="utf-8")

    print("Wrote", tbl_path, "and", summ_path)


if __name__ == "__main__":
    main()
