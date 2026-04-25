"""Check three things on Prosperity4 ROUND_3:
  1) HYDROGEL_PACK vs VELVETFRUIT_EXTRACT correlation -> are they independent?
  2) HYDROGEL_PACK mean-reversion around its level
  3) Realized vol of VELVETFRUIT_EXTRACT vs implied (~24%): variance-ratio sweep
  4) Deep-ITM VEV intrinsic-pinning check (4000, 4500)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Prosperity4Data/ROUND_3")
STEPS_PER_DAY = 10_000
TICK_SECONDS = 100  # one tick per 100 timestamps
SECONDS_PER_DAY = 1_000_000  # competition convention
ANNUAL_TICKS = 365 * STEPS_PER_DAY


def load(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
    return df


def wide_mid(day: int) -> pd.DataFrame:
    df = load(day)
    products = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", "VEV_4000", "VEV_4500"]
    df = df[df["product"].isin(products)]
    return df.pivot_table(index="timestamp", columns="product", values="mid_price").sort_index()


def main():
    panels = [wide_mid(d) for d in (0, 1, 2)]
    full = pd.concat(panels, ignore_index=False, axis=0)

    # ---- 1. Correlation of changes ----
    chg = full[["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]].diff().dropna()
    corr_chg = chg["HYDROGEL_PACK"].corr(chg["VELVETFRUIT_EXTRACT"])
    print(f"Corr( d HYDROGEL, d VELVETFRUIT ) = {corr_chg:+.4f}   (n={len(chg):,})")

    # ---- 2. Hydrogel mean-reversion characterization ----
    h = full["HYDROGEL_PACK"].dropna()
    print(f"\nHYDROGEL_PACK  mean={h.mean():.2f}  std={h.std():.2f}  min={h.min():.0f}  max={h.max():.0f}")
    # AR(1) on changes
    dh = h.diff().dropna()
    ar1 = dh.autocorr(lag=1)
    print(f"  d-HYDROGEL lag-1 autocorr = {ar1:+.4f}")
    # Z-score histogram around mean
    bands = [10, 15, 20, 25, 30]
    for b in bands:
        n_above = (h - h.mean() > b).sum()
        n_below = (h.mean() - h > b).sum()
        print(f"  |dev|>{b:>3}: above={n_above:>6}  below={n_below:>6}")

    # ---- 3. Velvet realized vol via variance-ratio across horizons ----
    s = full["VELVETFRUIT_EXTRACT"].dropna()
    print(f"\nVELVETFRUIT_EXTRACT  mean={s.mean():.2f}  std={s.std():.2f}")
    print("\nRealized annualized vol of VELVETFRUIT_EXTRACT log-returns at various lags:")
    print(f"{'lag':>6} {'vol_per_step':>14} {'ann_vol':>10} {'lag1_acf':>10}")
    logs = np.log(s.values)
    for lag in [1, 5, 10, 50, 100, 500, 1000, 5000]:
        if lag >= len(logs) // 2:
            break
        r = logs[lag:] - logs[:-lag]
        if r.size < 50:
            continue
        sigma_step = r.std() / math.sqrt(lag)  # per-step stdev
        ann = sigma_step * math.sqrt(ANNUAL_TICKS)
        # Lag-1 acf of return series at this aggregation
        if r.size > 5:
            acf1 = float(pd.Series(r).autocorr(lag=1))
        else:
            acf1 = float("nan")
        print(f"{lag:>6} {sigma_step:14.7f} {ann:10.4f} {acf1:10.4f}")

    # ---- 4. Deep-ITM intrinsic-pinning ----
    print("\nDeep-ITM check: average (mid - max(S-K, 0)) for K=4000, 4500")
    for k in (4000, 4500):
        col = f"VEV_{k}"
        if col not in full.columns:
            continue
        joined = full[["VELVETFRUIT_EXTRACT", col]].dropna()
        intrinsic = (joined["VELVETFRUIT_EXTRACT"] - k).clip(lower=0.0)
        diff = joined[col] - intrinsic
        print(f"  K={k}: mean_extrinsic={diff.mean():+.4f}  std={diff.std():.4f}  min={diff.min():.3f}  max={diff.max():.3f}  n={len(diff)}")


if __name__ == "__main__":
    main()
