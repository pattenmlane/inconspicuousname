#!/usr/bin/env python3
"""
Round 4 — **Rolling Sonic tight share** vs **forward extract mid** (K=20 tape steps).

For each day, at each timestamp in the joint inner-join panel, define:
  tight_share_W = fraction of past W ticks (inclusive) where joint gate was on.
Compare correlation of tight_share_W with fwd20_ext (same row).

Outputs: analysis_outputs/r4_rolling_tight_share_corr_k20.csv

W in {20, 50, 100} (tick steps; tape step = 100 time units in R4 prices).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
K_FWD = 20
WINDOWS = (20, 50, 100)


def load_px() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            if "day" not in df.columns:
                df["day"] = d
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def panel_day(px: pd.DataFrame, day: int) -> pd.DataFrame:
    def op(prod: str):
        v = (
            px[(px["day"] == day) & (px["product"] == prod)]
            .drop_duplicates(subset=["timestamp"], keep="first")
            .sort_values("timestamp")
        )
        bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
        mid = pd.to_numeric(v["mid_price"], errors="coerce")
        return pd.DataFrame(
            {
                "timestamp": v["timestamp"].astype(int),
                "spread": (ask - bid).astype(float),
                "mid": mid.astype(float),
            }
        )

    a = op("VEV_5200").rename(columns={"spread": "s5200"})
    b = op("VEV_5300").rename(columns={"spread": "s5300"})
    e = op("VELVETFRUIT_EXTRACT")[["timestamp", "mid"]].rename(columns={"mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(e, on="timestamp", how="inner")
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["tight"] = ((m["s5200"] <= TH) & (m["s5300"] <= TH)).astype(float)
    m["m_ext_f"] = m["m_ext"].shift(-K_FWD)
    m["fwd20"] = m["m_ext_f"] - m["m_ext"]
    return m


def main() -> int:
    px = load_px()
    rows = []
    for d in DAYS:
        m = panel_day(px, d)
        m = m.dropna(subset=["fwd20"])
        for W in WINDOWS:
            roll = m["tight"].rolling(window=W, min_periods=max(5, W // 4)).mean()
            c = float(roll.corr(m["fwd20"]))
            rows.append({"day": d, "window_ticks": W, "corr_roll_tight_share_fwd20": c, "n": len(m)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "r4_rolling_tight_share_corr_k20.csv", index=False)
    print(out.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
