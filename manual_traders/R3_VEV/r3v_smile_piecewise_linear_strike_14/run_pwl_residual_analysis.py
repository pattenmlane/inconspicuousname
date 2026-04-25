#!/usr/bin/env python3
"""
Round 3 tapes: PWL smile residual vs raw implied vol (Greek/IV thread).
For each subsampled row: T from csv_day + t_years_effective; IV_mkt from mid; knot IVs from
median bands; sigma_pwl(K); residual = IV_mkt - sigma_pwl. Also |call_pwl - mid| vs vega.
Outputs analysis_outputs/pwl_residual_stats.json
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _r3v_smile_core import (  # noqa: E402
    implied_vol_bisect,
    pwl_iv_strike,
    t_years_effective,
    bs_call_price,
    bs_vega,
)

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
KNOTS = (5000, 5200, 5400)
U = "VELVETFRUIT_EXTRACT"


def _median(xs: list[float]) -> float | None:
    ys = [x for x in xs if math.isfinite(x) and x > 0]
    if not ys:
        return None
    ys.sort()
    return ys[len(ys) // 2]


def knot_ivs_from_surface(strike_iv: dict[int, float | None]) -> tuple[float, float, float] | None:
    def ivs(ks: tuple[int, ...]) -> list[float]:
        out: list[float] = []
        for k in ks:
            v = strike_iv.get(k)
            if v is not None and math.isfinite(v) and v > 0:
                out.append(float(v))
        return out

    m0 = _median(ivs((5000, 5100)))
    m1 = _median(ivs((5100, 5200, 5300)))
    m2 = _median(ivs((5300, 5400, 5500)))
    if m0 is None or m1 is None or m2 is None:
        return None

    def clip(x: float) -> float:
        return max(0.04, min(3.5, x))

    return (clip(m0), clip(m1), clip(m2))


def main() -> None:
    out: dict = {
        "method": "IV from mid via bisection (r=0); T=t_years_effective(csv_day, ts); "
        "knot IVs = same median bands as trader; sigma_pwl(K); residual_iv = IV_mkt - sigma_pwl.",
        "by_day_strike": {},
    }
    step = 40
    for csv_day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{csv_day}.csv", sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        if U not in pvt.columns:
            continue
        rows = list(pvt.iloc[::step].iterrows())
        res_iv: dict[int, list[float]] = {k: [] for k in STRIKES}
        res_px: dict[int, list[float]] = {k: [] for k in STRIKES}
        vega_list: dict[int, list[float]] = {k: [] for k in STRIKES}

        for ts, row in rows:
            S = float(row[U])
            if not math.isfinite(S) or S <= 0:
                continue
            T = t_years_effective(csv_day, int(ts))
            strike_iv: dict[int, float | None] = {}
            for k, v in zip(STRIKES, VOUCHERS):
                if v not in row.index or pd.isna(row[v]):
                    strike_iv[k] = None
                    continue
                mid = float(row[v])
                strike_iv[k] = implied_vol_bisect(mid, S, float(k), T)
            knots = knot_ivs_from_surface(strike_iv)
            if knots is None:
                continue
            for k in STRIKES:
                iv_m = strike_iv.get(k)
                if iv_m is None or not math.isfinite(iv_m):
                    continue
                sig = pwl_iv_strike(float(k), KNOTS, knots)
                res_iv[k].append(iv_m - sig)
                mid = float(row[f"VEV_{k}"])
                theo = bs_call_price(S, float(k), T, sig)
                res_px[k].append(abs(mid - theo))
                vega_list[k].append(bs_vega(S, float(k), T, sig))

        out["by_day_strike"][str(csv_day)] = {}
        for k in STRIKES:
            if not res_iv[k]:
                continue
            out["by_day_strike"][str(csv_day)][str(k)] = {
                "n": len(res_iv[k]),
                "mean_abs_iv_residual": float(statistics.mean(abs(x) for x in res_iv[k])),
                "median_abs_iv_residual": float(statistics.median(abs(x) for x in res_iv[k])),
                "mean_abs_price_err": float(statistics.mean(res_px[k])) if res_px[k] else None,
                "median_vega": float(statistics.median(vega_list[k])) if vega_list[k] else None,
            }

    od = Path(__file__).resolve().parent / "analysis_outputs" / "pwl_residual_stats.json"
    od.parent.mkdir(parents=True, exist_ok=True)
    od.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", od)


if __name__ == "__main__":
    main()
