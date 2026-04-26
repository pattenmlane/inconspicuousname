#!/usr/bin/env python3
"""
Joint Sonic gate *persistence* vs extract forward move on the Phase-3 inner-join panel.

At timestamps where joint tight NOW, compare extract fwd20 (shift -20 on m_ext in aligned
panel) when joint tight STILL holds 20 steps forward vs when it has flipped to loose.
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
    rows = []
    for d in (1, 2, 3):
        m = aligned_panel(d)
        m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
        m["fwd_k"] = m["m_ext"].shift(-K) - m["m_ext"]
        m["tight_fwd"] = m["tight"].shift(-K).astype("boolean")
        sub = m[m["tight"] & m["fwd_k"].notna() & m["tight_fwd"].notna()]
        stay = sub.loc[sub["tight_fwd"] == True, "fwd_k"].to_numpy()
        leave = sub.loc[sub["tight_fwd"] == False, "fwd_k"].to_numpy()
        rows.append(
            {
                "csv_day": d,
                "n_tight_origin": int(len(sub)),
                "n_still_tight_k20": int(len(stay)),
                "n_flipped_loose_k20": int(len(leave)),
                "mean_fwd_still_tight": float(np.mean(stay)) if len(stay) else None,
                "mean_fwd_flipped_loose": float(np.mean(leave)) if len(leave) else None,
                "welch_stay_vs_leave": welch(stay, leave),
            }
        )

    pool_stay: list[float] = []
    pool_leave: list[float] = []
    for d in (1, 2, 3):
        m = aligned_panel(d)
        m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
        m["fwd_k"] = m["m_ext"].shift(-K) - m["m_ext"]
        m["tight_fwd"] = m["tight"].shift(-K).astype("boolean")
        sub = m[m["tight"] & m["fwd_k"].notna() & m["tight_fwd"].notna()]
        pool_stay.extend(sub.loc[sub["tight_fwd"] == True, "fwd_k"].tolist())
        pool_leave.extend(sub.loc[sub["tight_fwd"] == False, "fwd_k"].tolist())

    out = {
        "definition": "Inner-join panel rows; origin = joint tight; fwd20 = m_ext.shift(-20)-m_ext; tight_fwd = tight.shift(-20)",
        "per_day": rows,
        "pooled": {
            "n_still_tight": len(pool_stay),
            "n_flipped_loose": len(pool_leave),
            "welch_stay_vs_leave": welch(np.array(pool_stay), np.array(pool_leave)),
        },
    }
    pth = _OUT / "r4_joint_gate_persist_extract_fwd20.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
