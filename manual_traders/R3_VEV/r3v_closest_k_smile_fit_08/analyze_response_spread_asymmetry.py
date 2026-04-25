#!/usr/bin/env python3
"""Underlying move propagation into voucher beta + spread response + asymmetry.

Outputs per strike:
- beta_up / beta_down from contemporaneous dV/dS buckets
- lag-1 beta (dV_t vs dS_{t-1})
- spread widening after |dS| shocks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "response_spread_asymmetry.json"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)


def _lin_beta(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 20:
        return None
    den = float((x * x).sum())
    if abs(den) < 1e-12:
        return None
    return float((x * y).sum() / den)


def analyze_day(day: int) -> dict:
    df = pd.read_csv(
        REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv",
        sep=";",
    )
    pvt_mid = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
    pvt_spr = df.assign(spread=df["ask_price_1"] - df["bid_price_1"]).pivot_table(
        index="timestamp", columns="product", values="spread", aggfunc="first"
    )
    s = pvt_mid["VELVETFRUIT_EXTRACT"].astype(float)
    ds = s.diff()

    out: dict = {}
    shock_thr = float(ds.abs().quantile(0.90))

    for k in STRIKES:
        sym = f"VEV_{k}"
        if sym not in pvt_mid.columns:
            continue
        v = pvt_mid[sym].astype(float)
        dv = v.diff()
        sp = pvt_spr[sym].astype(float)
        dsp = sp.diff()

        m = pd.DataFrame({"ds": ds, "dv": dv, "sp": sp, "dsp": dsp}).dropna()
        if m.empty:
            continue
        ds_arr = m["ds"].to_numpy()
        dv_arr = m["dv"].to_numpy()
        up = ds_arr > 0
        dn = ds_arr < 0

        beta_all = _lin_beta(ds_arr, dv_arr)
        beta_up = _lin_beta(ds_arr[up], dv_arr[up]) if up.any() else None
        beta_dn = _lin_beta(ds_arr[dn], dv_arr[dn]) if dn.any() else None

        lag = pd.DataFrame({"ds_l1": ds.shift(1), "dv": dv}).dropna()
        beta_lag1 = _lin_beta(lag["ds_l1"].to_numpy(), lag["dv"].to_numpy())

        shock = m["ds"].abs() >= shock_thr
        spread_after_shock = float(m.loc[shock, "sp"].mean()) if shock.any() else None
        spread_normal = float(m.loc[~shock, "sp"].mean()) if (~shock).any() else None

        out[str(k)] = {
            "beta_all": beta_all,
            "beta_up": beta_up,
            "beta_down": beta_dn,
            "beta_up_minus_down": (beta_up - beta_dn) if beta_up is not None and beta_dn is not None else None,
            "beta_lag1": beta_lag1,
            "spread_mean_after_shock": spread_after_shock,
            "spread_mean_normal": spread_normal,
            "spread_widen_after_shock": (spread_after_shock - spread_normal)
            if spread_after_shock is not None and spread_normal is not None
            else None,
        }
    return {"shock_abs_ds_p90": shock_thr, "per_strike": out}


def main() -> None:
    out = {
        "method": "Per day, use voucher/underlying mid diffs. Betas are OLS through origin (dV~beta*dS), split by dS sign. Spread widening compares mean spread in |dS|>=p90 shocks vs normal.",
        "by_day": {str(d): analyze_day(d) for d in (0, 1, 2)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
