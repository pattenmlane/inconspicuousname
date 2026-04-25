#!/usr/bin/env python3
"""ATM implied vol vs extract moves: lead/lag and shock conditioning (Round 3 tapes)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "atm_iv_dS_lead_lag.json"

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
ATM_KS = (5100, 5200, 5300)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(s: float, k: float, t: float, sig: float) -> float:
    if t <= 1e-12 or sig <= 1e-12:
        return max(s - k, 0.0)
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / v
    d2 = d1 - v
    return s * norm_cdf(d1) - k * norm_cdf(d2)


def iv_bisect(px: float, s: float, k: float, t: float) -> float | None:
    if px <= max(s - k, 0.0) + 1e-6 or px >= s - 1e-6:
        return None
    lo, hi = 1e-4, 12.0
    if bs_call(s, k, t, lo) - px > 0 or bs_call(s, k, t, hi) - px < 0:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bs_call(s, k, t, mid) >= px:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def dte_eff(day: int, ts: int) -> float:
    return max(8.0 - float(day) - (int(ts) // 100) / 10_000.0, 1e-6)


def main() -> None:
    all_ds: list[float] = []
    all_div: list[float] = []
    by_day: dict = {}
    for day in (0, 1, 2):
        df = pd.read_csv(
            REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv",
            sep=";",
        )
        pvt = df.pivot_table(
            index="timestamp", columns="product", values="mid_price", aggfunc="first"
        )
        idx = pvt.index.to_list()
        s_arr = pvt["VELVETFRUIT_EXTRACT"].values.astype(float)
        # median IV across ATM cluster K
        ivs: list[np.ndarray] = []
        for k in ATM_KS:
            sym = f"VEV_{k}"
            if sym not in pvt.columns:
                continue
            row_iv = []
            for i, ts in enumerate(idx):
                t = dte_eff(day, int(ts)) / 365.0
                sv = float(s_arr[i])
                px = float(pvt.at[ts, sym]) if not pd.isna(pvt.at[ts, sym]) else float("nan")
                if pd.isna(px) or sv <= 0:
                    row_iv.append(float("nan"))
                else:
                    iv = iv_bisect(px, sv, float(k), t)
                    row_iv.append(iv if iv is not None else float("nan"))
            ivs.append(np.array(row_iv, dtype=float))
        if not ivs:
            continue
        stack = np.vstack(ivs)
        iv_m = np.nanmedian(stack, axis=0)
        ds = np.abs(np.diff(s_arr, prepend=s_arr[0]))
        div = np.abs(np.diff(iv_m, prepend=iv_m[0]))
        all_ds.extend(ds.tolist())
        all_div.extend(div.tolist())
        p90 = float(np.quantile(ds, 0.9))
        shock = ds >= p90
        by_day[str(day)] = {
            "median_abs_dS": float(np.median(ds)),
            "p90_abs_dS": p90,
            "median_abs_dIV_atm_median3": float(np.median(div)),
            "mean_div_when_shock": float(np.nanmean(div[shock])),
            "mean_div_when_calm": float(np.nanmean(div[~shock])),
        }
    a = np.array(all_ds, dtype=float)
    b = np.array(all_div, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b >= 0)
    rho = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 5 else 0.0
    out = {
        "method": "ATM cluster K in {5100,5200,5300}: each tick IV from bisection, take median. dS=|diff extract mid|, dIV=|diff median IV|. Pool days 0-2 for global corr.",
        "by_day": by_day,
        "pooled": {
            "n": int(m.sum()),
            "corr_absdS_absdIV": rho,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
