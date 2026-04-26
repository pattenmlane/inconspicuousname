#!/usr/bin/env python3
"""
Phase 2 — cross-instrument lead/lag: per csv day, inner-join EXTRACT, VEV_5200, VEV_5300
mids; d = first difference. corr( d_ext[t], d_5200[t-k] ) and same for 5300, k in lags
(snapshot index, same as Phase-1 K).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
_DATA = Path("Prosperity4Data/ROUND_4")
EXTRACT = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
LAGS = (0, 1, 2, 3, 5, 10, 20)


def one_sym(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    v = (
        df[df["product"] == sym]
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
    )
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(mid=mid)[["timestamp", "mid"]].copy()


def panel_day(d: int) -> pd.DataFrame:
    df = pd.read_csv(_DATA / f"prices_round_4_day_{d}.csv", sep=";")
    df = df[df["day"] == d]
    e = one_sym(df, EXTRACT).rename(columns={"mid": "m_e"})
    a = one_sym(df, V5200).rename(columns={"mid": "m_52"})
    b = one_sym(df, V5300).rename(columns={"mid": "m_53"})
    return e.merge(a, on="timestamp", how="inner").merge(b, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)


def corr_lag(du: np.ndarray, dv: np.ndarray, k: int) -> float | None:
    if k < 0 or len(du) != len(dv):
        return None
    if k == 0:
        x, y = du, dv
    else:
        if len(du) <= k:
            return None
        x = du[k:]
        y = dv[:-k]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return None
    return float(np.corrcoef(x[m], y[m])[0, 1])


def main() -> None:
    out: dict = {"method": "corr( d_ext[t], d_vev[t-k] ); d = diff along time within day", "lags": list(LAGS), "per_day": {}}
    for d in (1, 2, 3):
        m = panel_day(d)
        m["d_e"] = m["m_e"].diff()
        m["d_52"] = m["m_52"].diff()
        m["d_53"] = m["m_53"].diff()
        u = m["d_e"].to_numpy()
        v52 = m["d_52"].to_numpy()
        v53 = m["d_53"].to_numpy()
        out["per_day"][str(d)] = {
            str(k): {
                "corr_d_ext_d5200": corr_lag(u, v52, k),
                "corr_d_ext_d5300": corr_lag(u, v53, k),
            }
            for k in LAGS
        }

    pth = OUT / "r4_leadlag_extract_vev_mids_deltas_by_day.json"
    pth.write_text(json.dumps(out, indent=2))
    print(pth)


if __name__ == "__main__":
    main()
