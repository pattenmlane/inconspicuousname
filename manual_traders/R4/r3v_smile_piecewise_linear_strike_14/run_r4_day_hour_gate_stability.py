#!/usr/bin/env python3
"""
Per (csv_day, hour): Welch tight vs loose on extract fwd20 (aligned panel, K=20).
Tests whether pooled-hour significance (iteration 7) holds day-by-day.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
_DATA = Path("Prosperity4Data/ROUND_4")
_OUT = ROOT / "analysis_outputs"
_OUT.mkdir(parents=True, exist_ok=True)

VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"
TH = 2
K = 20
MIN_TIGHT = 10
MIN_LOOSE = 25


def _one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    v = v.assign(spread=(ask - bid).astype(float), mid=mid)
    return v[["timestamp", "spread", "mid"]].copy()


def aligned_panel(csv_day: int) -> pd.DataFrame:
    df = pd.read_csv(_DATA / f"prices_round_4_day_{csv_day}.csv", sep=";")
    df = df[df["day"] == csv_day]
    a = _one_product(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = _one_product(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = _one_product(df, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    m["csv_day"] = csv_day
    return m.sort_values("timestamp").reset_index(drop=True)


def welch(a: np.ndarray, b: np.ndarray) -> dict | None:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    t = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return {
        "mean_tight": float(np.mean(a)),
        "mean_loose": float(np.mean(b)),
        "n_tight": int(len(a)),
        "n_loose": int(len(b)),
        "t_stat": float(t.statistic),
        "p_value": float(t.pvalue),
    }


def main() -> None:
    cells: dict[tuple[int, int], dict | None] = {}
    for d in (1, 2, 3):
        m = aligned_panel(d)
        m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
        m["fwd_k"] = m["m_ext"].shift(-K) - m["m_ext"]
        m["hour"] = (m["timestamp"] // 10000) % 24
        pv = m[m["fwd_k"].notna()]
        for h in range(24):
            sub = pv[pv["hour"] == h]
            ta = sub.loc[sub["tight"], "fwd_k"].values
            lo = sub.loc[~sub["tight"], "fwd_k"].values
            if len(ta) < MIN_TIGHT or len(lo) < MIN_LOOSE:
                cells[(d, h)] = None
            else:
                cells[(d, h)] = welch(ta, lo)

    # Summary: for each hour, count days where tight>loose and p<0.1
    rollup: dict[str, dict] = {}
    for h in range(24):
        ok_dir = 0
        ok_sig = 0
        n_valid = 0
        for d in (1, 2, 3):
            w = cells.get((d, h))
            if w is None:
                continue
            n_valid += 1
            if w["mean_tight"] > w["mean_loose"]:
                ok_dir += 1
            if w["mean_tight"] > w["mean_loose"] and w["p_value"] < 0.1:
                ok_sig += 1
        rollup[str(h)] = {
            "n_days_with_enough_n": n_valid,
            "days_tight_mean_gt_loose": ok_dir,
            "days_tight_gt_loose_and_p_lt_0.1": ok_sig,
        }

    out = {
        "min_counts": {"tight": MIN_TIGHT, "loose": MIN_LOOSE},
        "cells": {
            f"day{d}_hour{h}": cells.get((d, h)) for d in (1, 2, 3) for h in range(24)
        },
        "rollup_by_hour": rollup,
    }
    # trim None keys for readability
    out["cells"] = {k: v for k, v in out["cells"].items() if v is not None}
    pth = _OUT / "r4_day_hour_gate_welch_fwd20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
