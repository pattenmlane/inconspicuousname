#!/usr/bin/env python3
"""
Leave-one-csv-day-out: pooled Sonic gate panel (5200+5300+extract) Welch tight vs loose
on extract fwd20, recomputed leaving out each day in turn — tests stability of the signal
when one day is withheld from estimation sample.
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
# Reuse phase3 helpers by import - duplicate minimal logic to avoid circular import
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"
TH = 2
K = 20


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


def add_gate_fwd(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["tight"] = (out["s5200"] <= TH) & (out["s5300"] <= TH)
    out["fwd_k"] = out["m_ext"].shift(-K) - out["m_ext"]
    return out


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b)}
    t = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return {
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "t_stat": float(t.statistic),
        "p_value": float(t.pvalue),
    }


def main() -> None:
    panels = {d: add_gate_fwd(aligned_panel(d)) for d in (1, 2, 3)}
    pool = pd.concat(panels.values(), ignore_index=True)
    pv = pool[pool["fwd_k"].notna()]

    out: dict = {"loo_welch_exclude_day": {}, "per_day_only": {}}

    for ex in (1, 2, 3):
        sub = pv[pv["csv_day"] != ex]
        t_mask = sub["tight"]
        out["loo_welch_exclude_day"][str(ex)] = welch(
            sub.loc[t_mask, "fwd_k"].values,
            sub.loc[~t_mask, "fwd_k"].values,
        )

    for d in (1, 2, 3):
        p = panels[d]
        v = p[p["fwd_k"].notna()]
        tm = v["tight"]
        out["per_day_only"][str(d)] = welch(
            v.loc[tm, "fwd_k"].values,
            v.loc[~tm, "fwd_k"].values,
        )

    out["sign_stability"] = {
        "all_three_days_tight_mean_gt_loose": all(
            out["per_day_only"][str(d)]["mean_a"] > out["per_day_only"][str(d)]["mean_b"]
            for d in (1, 2, 3)
            if "mean_a" in out["per_day_only"][str(d)]
        ),
        "all_three_loo_t_stat_positive": all(
            out["loo_welch_exclude_day"][str(ex)]["t_stat"] > 0
            for ex in (1, 2, 3)
            if "t_stat" in out["loo_welch_exclude_day"][str(ex)]
        ),
    }

    pth = _OUT / "r4_loo_joint_gate_welch_fwd20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
