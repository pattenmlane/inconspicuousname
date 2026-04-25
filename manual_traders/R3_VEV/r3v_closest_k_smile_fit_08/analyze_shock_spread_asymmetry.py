#!/usr/bin/env python3
"""Shock rows: spread level and |dV| asymmetry when dU up vs down.

For each Round3 day and VEV strike: among rows with |dU| >= p90 (shock), compare
mean top-of-book spread with calm; on shock rows only, OLS beta dV~dU for dU>0 vs dU<0.
Pooled summary across days for decision support (wider books on shocks, put/call asymmetry).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "shock_spread_asymmetry.json"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)


def _beta_origin(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 15:
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
    pvt_mid = df.pivot_table(
        index="timestamp", columns="product", values="mid_price", aggfunc="first"
    )
    pvt_spr = df.assign(spread=df["ask_price_1"] - df["bid_price_1"]).pivot_table(
        index="timestamp", columns="product", values="spread", aggfunc="first"
    )
    s = pvt_mid["VELVETFRUIT_EXTRACT"].astype(float)
    ds = s.diff()
    thr = float(ds.abs().quantile(0.90))
    shock = ds.abs() >= thr

    out: dict = {}
    for k in STRIKES:
        sym = f"VEV_{k}"
        if sym not in pvt_mid.columns:
            continue
        v = pvt_mid[sym].astype(float)
        dv = v.diff()
        sp = pvt_spr[sym].astype(float)
        m = pd.DataFrame({"ds": ds, "dv": dv, "sp": sp}).dropna()
        if m.empty:
            continue
        calm = m["ds"].abs() < thr
        sh = m["ds"].abs() >= thr
        spread_calm = float(m.loc[calm, "sp"].mean()) if calm.any() else None
        spread_shock = float(m.loc[sh, "sp"].mean()) if sh.any() else None
        spread_diff = (
            (spread_shock - spread_calm)
            if spread_shock is not None and spread_calm is not None
            else None
        )

        ms = m.loc[sh]
        if len(ms) < 20:
            out[str(k)] = {
                "shock_abs_du_p90": thr,
                "spread_mean_calm": spread_calm,
                "spread_mean_shock": spread_shock,
                "spread_shock_minus_calm": spread_diff,
                "beta_shock_up": None,
                "beta_shock_down": None,
                "beta_up_minus_down": None,
            }
            continue
        up = ms["ds"].to_numpy() > 0
        dn = ms["ds"].to_numpy() < 0
        bu = _beta_origin(ms.loc[up, "ds"].to_numpy(), ms.loc[up, "dv"].to_numpy()) if up.any() else None
        bd = _beta_origin(ms.loc[dn, "ds"].to_numpy(), ms.loc[dn, "dv"].to_numpy()) if dn.any() else None
        out[str(k)] = {
            "shock_abs_du_p90": thr,
            "spread_mean_calm": spread_calm,
            "spread_mean_shock": spread_shock,
            "spread_shock_minus_calm": spread_diff,
            "beta_shock_up": bu,
            "beta_shock_down": bd,
            "beta_up_minus_down": (bu - bd) if bu is not None and bd is not None else None,
        }
    return out


def main() -> None:
    by_day = {str(d): analyze_day(d) for d in (0, 1, 2)}
    pooled: dict = {}
    keys = set()
    for dct in by_day.values():
        keys |= set(dct.keys())
    for k in sorted(keys, key=int):
        sp_d: list[float] = []
        bu: list[float] = []
        bd: list[float] = []
        for dct in by_day.values():
            if k not in dct:
                continue
            row = dct[k]
            if row.get("spread_shock_minus_calm") is not None:
                sp_d.append(row["spread_shock_minus_calm"])
            if row.get("beta_shock_up") is not None:
                bu.append(row["beta_shock_up"])
            if row.get("beta_shock_down") is not None:
                bd.append(row["beta_shock_down"])
        pooled[k] = {
            "median_spread_shock_minus_calm_across_days": float(np.median(sp_d)) if sp_d else None,
            "median_beta_shock_up_across_days": float(np.median(bu)) if bu else None,
            "median_beta_shock_down_across_days": float(np.median(bd)) if bd else None,
        }
    out = {
        "method": "Shock = |d extract mid| >= day p90. Calm = rest. Compare mean VEV spread; on shock rows only, OLS through origin for dV vs dU in up vs down buckets.",
        "by_day": by_day,
        "pooled_median_by_strike": pooled,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
