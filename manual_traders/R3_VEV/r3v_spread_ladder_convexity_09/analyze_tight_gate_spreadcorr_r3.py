#!/usr/bin/env python3
"""
Supplemental stats for round3work/vouchers_final_strategy: multi-day P(tight),
corr(s5200,s5300), and mean K-step forward extract mid when TH=2, K=20.
Uses Prosperity4Data tapes only (pandas); no separate tipworkflow import.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # .../workspace
DATA = ROOT / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_tight_gate_multiday_v22.json"

U = "VELVETFRUIT_EXTRACT"
G52, G53 = "VEV_5200", "VEV_5300"
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
    v = v.assign(s=(ask - bid).astype(float), m_ext=mid)
    return v[["timestamp", "s", "m_ext"]].copy()


def aligned_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    a = _one_product(df, G52).rename(columns={"s": "s5200", "m_ext": "mid5200"})
    b = _one_product(df, G53).rename(columns={"s": "s5300", "m_ext": "mid5300"})
    e = _one_product(df, U).rename(columns={"s": "s_ext", "m_ext": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    return m.sort_values("timestamp").reset_index(drop=True)


def day_stats(m: pd.DataFrame, day: int) -> dict:
    s52 = m["s5200"].to_numpy()
    s53 = m["s5300"].to_numpy()
    n = len(m)
    if n < 30:
        return {"day": day, "n_merged": int(n), "note": "too_short"}
    c = float(np.corrcoef(s52, s53)[0, 1]) if n > 1 else 0.0
    tight = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m2 = m.copy()
    m2["tight"] = tight
    m2["fwd"] = m2["m_ext"].shift(-K) - m2["m_ext"]
    sub = m2.dropna(subset=["fwd"])
    tmask = sub["tight"]
    t = sub.loc[tmask, "fwd"].to_numpy()
    nt = sub.loc[~tmask, "fwd"].to_numpy()
    p_tight = float(tmask.mean()) if len(sub) else 0.0
    return {
        "day": day,
        "n_merged": n,
        "corr_s5200_s5300": round(c, 6),
        "K": K,
        "TH": TH,
        "P_tight": round(p_tight, 5),
        "valid_fwd_n": int(len(sub)),
        "mean_fwd_mid_tight": round(float(np.mean(t)), 6) if len(t) else None,
        "mean_fwd_mid_not_tight": round(float(np.mean(nt)), 6) if len(nt) else None,
    }


def main() -> None:
    days: list[dict] = []
    for p in sorted(DATA.glob("prices_round_3_day_*.csv")):
        name = p.stem
        try:
            d = int(name.split("_")[-1])
        except ValueError:
            continue
        m = aligned_panel(p)
        days.append(day_stats(m, d))
    doc = {
        "source": "Prosperity4Data/ROUND_3/prices_round_3_day_*.csv (semicolon)",
        "method": "inner join 5200+5300+extract on timestamp; spread=ask1-bid1; tight if both spread<=TH; forward extract mid = m_ext(t+K)-m_ext(t)",
        "TH": TH,
        "K": K,
        "days": days,
    }
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
