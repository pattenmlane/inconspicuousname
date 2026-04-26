#!/usr/bin/env python3
"""
Hour-of-day × joint gate: on aligned inner-join panel (5200,5300,extract), for each hour
(ts//10000)%24, mean extract fwd20 when tight vs loose and sample sizes.

Phase-3 style panel and fwd20; regime split for stability across hours.
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
    panels = []
    for d in (1, 2, 3):
        m = aligned_panel(d)
        m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
        m["fwd_k"] = m["m_ext"].shift(-K) - m["m_ext"]
        m["hour"] = (m["timestamp"] // 10000) % 24
        panels.append(m)
    pool = pd.concat(panels, ignore_index=True)
    pv = pool[pool["fwd_k"].notna()].copy()

    by_hour: dict[str, dict] = {}
    for h in range(24):
        sub = pv[pv["hour"] == h]
        if sub.empty:
            continue
        ta = sub.loc[sub["tight"], "fwd_k"].values
        lo = sub.loc[~sub["tight"], "fwd_k"].values
        w = welch(ta, lo)
        if w is None:
            continue
        by_hour[str(h)] = w

    out = {
        "method": "hour=(timestamp//10000)%24; tight=(s5200<=2)&(s5300<=2); fwd20 on m_ext",
        "by_hour_welch_tight_vs_loose": by_hour,
        "hours_with_both_ns_ge_30": [
            h
            for h, v in by_hour.items()
            if v.get("n_tight", 0) >= 30 and v.get("n_loose", 0) >= 30
        ],
    }
    pth = _OUT / "r4_hour_joint_gate_extract_fwd20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
