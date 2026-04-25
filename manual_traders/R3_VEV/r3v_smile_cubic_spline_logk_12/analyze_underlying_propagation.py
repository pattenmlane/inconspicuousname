#!/usr/bin/env python3
"""Analyze how VELVETFRUIT_EXTRACT moves propagate to each VEV voucher.
Outputs:
- analysis_outputs/underlying_propagation_by_strike.csv
- analysis_outputs/underlying_propagation_shock_regimes.csv
Includes:
  * contemporaneous beta of voucher mid changes vs extract changes
  * lag-1 beta and correlation
  * elasticity (dVoucher / dExtract) by strike/day
  * spread widening conditioned on large up/down extract shocks
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
EX = "VELVETFRUIT_EXTRACT"
VEVS = [
    "VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500",
]


def f(v: str) -> float | None:
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_day(day: int):
    path = DATA / f"prices_round_3_day_{day}.csv"
    rows = list(csv.DictReader(path.open(), delimiter=";"))
    by_ts: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for r in rows:
        by_ts[int(r["timestamp"])][r["product"]] = r
    return by_ts


def beta(x: list[float], y: list[float]) -> float | None:
    if len(x) < 5:
        return None
    xx = np.array(x)
    yy = np.array(y)
    vx = float(np.var(xx))
    if vx <= 1e-12:
        return None
    return float(np.cov(xx, yy, ddof=0)[0, 1] / vx)


def corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 5:
        return None
    xx = np.array(x)
    yy = np.array(y)
    sx = float(np.std(xx))
    sy = float(np.std(yy))
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return float(np.corrcoef(xx, yy)[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    by_strike_rows = []
    regime_rows = []

    for day in (0, 1, 2):
        by_ts = load_day(day)
        ts_sorted = sorted(by_ts.keys())

        ex_mid = []
        for ts in ts_sorted:
            r = by_ts[ts].get(EX)
            if not r:
                ex_mid.append(None)
                continue
            ex_mid.append(f(r["mid_price"]))

        dex = []
        for i in range(1, len(ts_sorted)):
            if ex_mid[i] is None or ex_mid[i - 1] is None:
                dex.append(None)
            else:
                dex.append(ex_mid[i] - ex_mid[i - 1])

        # Shock thresholds by day from |dExtract|
        abs_d = [abs(v) for v in dex if v is not None]
        q95 = float(np.quantile(abs_d, 0.95)) if abs_d else 0.0

        for sym in VEVS:
            v_mid = []
            spread = []
            for ts in ts_sorted:
                r = by_ts[ts].get(sym)
                if not r:
                    v_mid.append(None)
                    spread.append(None)
                    continue
                m = f(r["mid_price"])
                b = f(r["bid_price_1"])
                a = f(r["ask_price_1"])
                v_mid.append(m)
                spread.append((a - b) if (a is not None and b is not None) else None)

            dv = []
            for i in range(1, len(ts_sorted)):
                if v_mid[i] is None or v_mid[i - 1] is None:
                    dv.append(None)
                else:
                    dv.append(v_mid[i] - v_mid[i - 1])

            x0, y0 = [], []
            xlag, ylag = [], []
            elas = []
            up_sp, dn_sp, all_sp = [], [], []

            for i in range(1, len(ts_sorted)):
                dx = dex[i - 1]
                dvi = dv[i - 1]
                if spread[i] is not None:
                    all_sp.append(spread[i])
                if dx is not None and dvi is not None:
                    x0.append(dx)
                    y0.append(dvi)
                    if abs(dx) > 1e-9:
                        elas.append(dvi / dx)
                    if abs(dx) >= q95 and spread[i] is not None:
                        if dx > 0:
                            up_sp.append(spread[i])
                        elif dx < 0:
                            dn_sp.append(spread[i])
                # lag-1 response
                if i >= 2:
                    dx_lag = dex[i - 2]
                    if dx_lag is not None and dvi is not None:
                        xlag.append(dx_lag)
                        ylag.append(dvi)

            b0 = beta(x0, y0)
            c0 = corr(x0, y0)
            b1 = beta(xlag, ylag)
            c1 = corr(xlag, ylag)
            med_elas = float(np.median(elas)) if elas else None

            by_strike_rows.append(
                {
                    "day": day,
                    "symbol": sym,
                    "n_obs": len(x0),
                    "beta_contemp": b0,
                    "corr_contemp": c0,
                    "beta_lag1": b1,
                    "corr_lag1": c1,
                    "median_elasticity": med_elas,
                    "q95_abs_dextract": q95,
                }
            )

            regime_rows.append(
                {
                    "day": day,
                    "symbol": sym,
                    "mean_spread_all": float(np.mean(all_sp)) if all_sp else None,
                    "mean_spread_shock_up": float(np.mean(up_sp)) if up_sp else None,
                    "mean_spread_shock_down": float(np.mean(dn_sp)) if dn_sp else None,
                    "n_shock_up": len(up_sp),
                    "n_shock_down": len(dn_sp),
                }
            )

    p1 = OUT / "underlying_propagation_by_strike.csv"
    with p1.open("w", newline="", encoding="utf-8") as fobj:
        w = csv.DictWriter(
            fobj,
            fieldnames=[
                "day",
                "symbol",
                "n_obs",
                "beta_contemp",
                "corr_contemp",
                "beta_lag1",
                "corr_lag1",
                "median_elasticity",
                "q95_abs_dextract",
            ],
        )
        w.writeheader()
        w.writerows(by_strike_rows)

    p2 = OUT / "underlying_propagation_shock_regimes.csv"
    with p2.open("w", newline="", encoding="utf-8") as fobj:
        w = csv.DictWriter(
            fobj,
            fieldnames=[
                "day",
                "symbol",
                "mean_spread_all",
                "mean_spread_shock_up",
                "mean_spread_shock_down",
                "n_shock_up",
                "n_shock_down",
            ],
        )
        w.writeheader()
        w.writerows(regime_rows)

    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
