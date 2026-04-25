#!/usr/bin/env python3
"""Deeper contract-level analysis: |dU| shock conditioning on *|dV|* and cross-corr dU vs dV at lags.

Pooled over Round3 days 0-2 (same tape as other scripts). Complements
response_spread_asymmetry (beta OLS) and atm_iv (ATM IV) by quantifying
whether voucher mids move in the same tick as extract or lag, and whether
|dV| spikes when |dU| is in the top decile.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "voucher_lead_lag_shock.json"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
LAGS = range(-3, 4)


def _xcorr(d_u: np.ndarray, d_v: np.ndarray) -> dict[str, float | None]:
    """Pearson corr between d_v[t] and d_u[t+lag] (lag>0: V leads U? actually d_u shifted)."""
    n = min(d_u.size, d_v.size)
    if n < 50:
        return {str(l): None for l in LAGS}
    u = d_u.astype(float)
    v = d_v.astype(float)
    out: dict[str, float | None] = {}
    for lag in LAGS:
        if lag == 0:
            uu, vv = u, v
        elif lag > 0:
            uu = u[lag:]
            vv = v[:-lag]
        else:
            k = -lag
            uu = u[:-k]
            vv = v[k:]
        m = (np.isfinite(uu) & np.isfinite(vv))
        uu, vv = uu[m], vv[m]
        if uu.size < 30:
            out[str(lag)] = None
            continue
        if uu.std() < 1e-9 or vv.std() < 1e-9:
            out[str(lag)] = None
        else:
            out[str(lag)] = float(np.corrcoef(uu, vv)[0, 1])
    return out


def analyze_pooled() -> dict:
    frames: list[pd.DataFrame] = []
    for day in (0, 1, 2):
        path = REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        if "VELVETFRUIT_EXTRACT" not in pvt.columns:
            continue
        s = pvt["VELVETFRUIT_EXTRACT"].astype(float)
        ds = s.diff()
        row = {"day": day, "ds": ds}
        for k in STRIKES:
            sym = f"VEV_{k}"
            if sym in pvt.columns:
                v = pvt[sym].astype(float)
                row[f"dv_{k}"] = v.diff()
        frames.append(pd.DataFrame(row))
    m = pd.concat(frames, ignore_index=False).dropna()
    m["abs_ds"] = m["ds"].abs()
    p90 = float(m["abs_ds"].quantile(0.90))
    shock = m["abs_ds"] >= p90

    per_strike: dict = {}
    for k in STRIKES:
        col = f"dv_{k}"
        if col not in m.columns:
            continue
        dv = m[col]
        ad = dv.abs()
        mean_abs_calm = float(ad.loc[~shock].mean()) if (~shock).any() else None
        mean_abs_shock = float(ad.loc[shock].mean()) if shock.any() else None
        ratio = (mean_abs_shock / mean_abs_calm) if mean_abs_calm and mean_abs_shock and mean_abs_calm > 1e-9 else None
        xcr = _xcorr(m["ds"].to_numpy(), dv.to_numpy())
        per_strike[str(k)] = {
            "p90_abs_dU_pooled": p90,
            "mean_abs_dV_calm": mean_abs_calm,
            "mean_abs_dV_shock": mean_abs_shock,
            "mean_abs_dV_shock_over_calm": ratio,
            "xcorr_dV_dU_lag": xcr,
        }

    return {
        "pooled_shock_threshold_abs_dU_p90": p90,
        "n_rows": int(len(m)),
        "per_strike": per_strike,
    }


def main() -> None:
    out = {
        "method": "Pooled days 0-1-2, same timestamps. Shock = |dU| >= pooled p90. Report mean |dV| shock vs calm, ratio, and Pearson corr(dU_t, dV_{t+lag}) for lag in -3..3 (lag 0: contemporaneous; lag +1: dV with previous tick dU).",
        "pooled": analyze_pooled(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
