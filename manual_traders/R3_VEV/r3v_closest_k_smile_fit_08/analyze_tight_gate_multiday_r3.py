#!/usr/bin/env python3
"""Multiday extension of round3work/vouchers_final_strategy: joint 5200/5300 gate, days 0..3.

Replicates STRATEGY.txt definitions: spread=ask1-bid1, tight=(s5200<=TH)&(s5300<=TH), inner join
with extract, K-step forward return on extract mid. Writes JSON for this branch; no plot deps.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "tight_gate_multiday_d0_3.json"

VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"
TH = 2
K = 20


def one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
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


def aligned_panel(df: pd.DataFrame) -> pd.DataFrame:
    a = one_product(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_product(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_product(df, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    return m.sort_values("timestamp").reset_index(drop=True)


def add_forward(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["tight"] = (out["s5200"] <= TH) & (out["s5300"] <= TH)
    out["m_ext_f"] = out["m_ext"].shift(-K)
    out["fwd_k"] = out["m_ext_f"] - out["m_ext"]
    return out


def per_day(path: Path, day_id: int) -> dict:
    df = pd.read_csv(path, sep=";")
    p = add_forward(aligned_panel(df))
    valid = p["fwd_k"].notna()
    pv = p.loc[valid]
    t = pv["tight"]
    ft = pv.loc[t, "fwd_k"]
    fn = pv.loc[~t, "fwd_k"]
    tstat, pval = (np.nan, np.nan)
    if len(ft) >= 2 and len(fn) >= 2:
        a = stats.ttest_ind(ft.values, fn.values, equal_var=False, nan_policy="omit")
        tstat, pval = float(a.statistic), float(a.pvalue)
    c_sp = float(pv["s5200"].corr(pv["s5300"]))
    return {
        "day_index_in_csv": int(df["day"].iloc[0]) if "day" in df.columns else int(day_id),
        "n_valid_fwd": int(len(pv)),
        "P_tight": float(t.mean()) if len(t) else 0.0,
        "mean_fwd_tight": float(ft.mean()) if len(ft) else None,
        "mean_fwd_nottight": float(fn.mean()) if len(fn) else None,
        "welch_tstat": tstat,
        "welch_p": pval,
        "corr_s5200_s5300": c_sp,
    }


def main() -> None:
    by_day: dict = {}
    for d in (0, 1, 2, 3):
        path = REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{d}.csv"
        if not path.exists():
            continue
        by_day[str(d)] = per_day(path, d)
    out = {
        "method": f"TH={TH}, K={K} forward steps on VELVETFRUIT_EXTRACT mid; inner-join 5200+5300+extract. Same as STRATEGY.txt / analyze_vev_5200_5300_tight_gate_r3.py.",
        "days": by_day,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
