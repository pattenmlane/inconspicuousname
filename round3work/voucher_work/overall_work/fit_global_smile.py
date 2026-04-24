"""
Pool implied vols across all VEV_* strikes and historical days 0–2; fit quadratic
  IV(m_t) with m_t = log(K/S) / sqrt(T_years)
matching FrankfurtHedgehogs_polished.get_iv m_t_k definition.

Writes fitted_smile_coeffs.json for use by 5200_work backtest / trader.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent.parent
_COMBINED = REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"
sys.path.insert(0, str(_COMBINED))

from plot_iv_smile_round3 import (  # noqa: E402
    VOUCHERS,
    implied_vol_call,
    load_day_wide,
    subsample_wide,
    t_years_effective,
)

OUT_JSON = Path(__file__).resolve().parent / "fitted_smile_coeffs.json"
STEP = 20


def main() -> None:
    xs: list[float] = []
    ys: list[float] = []
    for day in (0, 1, 2):
        wide = load_day_wide(day).sort_index()
        wsub = subsample_wide(wide, step=STEP)
        for ts, row in wsub.iterrows():
            ts_i = int(ts)
            S = float(row["S"])
            if S <= 0:
                continue
            T = t_years_effective(day, ts_i)
            if T <= 0:
                continue
            sqrtT = math.sqrt(T)
            for v in VOUCHERS:
                if v not in row.index:
                    continue
                mid = float(row[v])
                K = float(v.split("_")[1])
                iv = implied_vol_call(mid, S, K, T, 0.0)
                if not np.isfinite(iv):
                    continue
                m_t = math.log(K / S) / sqrtT
                if not np.isfinite(m_t):
                    continue
                xs.append(m_t)
                ys.append(iv)

    xf = np.asarray(xs, dtype=float)
    yf = np.asarray(ys, dtype=float)
    m = np.isfinite(xf) & np.isfinite(yf)
    xf, yf = xf[m], yf[m]
    if len(xf) < 100:
        raise SystemExit(f"too few points: {len(xf)}")

    coeff = np.polyfit(xf, yf, 2)
    resid = yf - np.polyval(coeff, xf)

    payload = {
        "coeffs_high_to_low": [float(c) for c in coeff],
        "description": "np.polyfit / np.poly1d order: m_t^2, m_t^1, constant; m_t=log(K/S)/sqrt(T)",
        "n_points": int(len(xf)),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mean_abs_resid": float(np.mean(np.abs(resid))),
        "subsample_step": STEP,
        "dte_calendar": {"csv_day_0": 8, "csv_day_1": 7, "csv_day_2": 6},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote", OUT_JSON, "n=", payload["n_points"], "rmse=", payload["rmse"])
    print("coeffs (poly1d order):", payload["coeffs_high_to_low"])


if __name__ == "__main__":
    main()
