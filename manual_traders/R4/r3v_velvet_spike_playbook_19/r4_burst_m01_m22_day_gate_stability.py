#!/usr/bin/env python3
"""
Round 4 follow-up — day-stability of Mark01→Mark22 multi-VEV bursts vs extract fwd20,
split by Sonic joint_tight and extract L1 spread bucket at burst timestamp.

Tape only (no sim). Bursts: same (day,timestamp) with >=3 VEV_* trades buyer=Mark 01,
seller=Mark 22. Forward extract mid: +K rows in prices tape (K=20).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH = 2
K = 20
DAYS = (1, 2, 3)


def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    df["product"] = df["product"].astype(str)
    return df


def one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(
        spread=(ask - bid).astype(float),
        mid=mid,
    )[["timestamp", "spread", "mid"]].copy()


def aligned_panel(day: int) -> pd.DataFrame:
    df = load_prices(day)
    a = one_product(df, "VEV_5200").rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_product(df, "VEV_5300").rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_product(df, "VELVETFRUIT_EXTRACT").rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["day"] = day
    m["joint_tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["fwd_k"] = m["m_ext"].shift(-K) - m["m_ext"]
    return m


def load_trades(day: int) -> pd.DataFrame:
    t = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
    t["day"] = day
    t["product"] = t["symbol"].astype(str)
    return t


def burst_timestamps(day: int, min_vev: int = 3) -> pd.DataFrame:
    tr = load_trades(day)
    m = tr[
        (tr["buyer"] == "Mark 01")
        & (tr["seller"] == "Mark 22")
        & tr["product"].str.startswith("VEV_", na=False)
    ]
    if m.empty:
        return pd.DataFrame(columns=["day", "timestamp", "n_vev"])
    g = m.groupby("timestamp", as_index=False).size()
    g = g.rename(columns={"size": "n_vev"})
    g = g[g["n_vev"] >= min_vev].copy()
    g["day"] = day
    return g[["day", "timestamp", "n_vev"]]


def spread_bucket(s: float) -> str:
    if pd.isna(s) or s <= 0:
        return "na"
    if s <= 2:
        return "tight"
    if s <= 6:
        return "mid"
    return "wide"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = {d: aligned_panel(d) for d in DAYS}
    rows = []
    raw_rows = []
    for d in DAYS:
        panel = panels[d].set_index("timestamp")
        bt = burst_timestamps(d)
        for _, r in bt.iterrows():
            ts = int(r["timestamp"])
            if ts not in panel.index:
                continue
            row = panel.loc[ts]
            fwd = float(row["fwd_k"]) if pd.notna(row["fwd_k"]) else float("nan")
            jt = bool(row["joint_tight"])
            sext = float(row["s_ext"])
            sb = spread_bucket(sext)
            raw_rows.append(
                {
                    "day": d,
                    "timestamp": ts,
                    "n_vev": int(r["n_vev"]),
                    "joint_tight": jt,
                    "s_ext": sext,
                    "s_ext_bucket": sb,
                    "fwd20_ex": fwd,
                }
            )
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(OUT / "r4_burst_m01_m22_rows_with_gate_fwd20.csv", index=False)

    if len(raw) == 0:
        return
    for (d, jt, sb), g in raw.groupby(["day", "joint_tight", "s_ext_bucket"]):
        v = g["fwd20_ex"].dropna()
        rows.append(
            {
                "day": int(d),
                "joint_tight": bool(jt),
                "s_ext_bucket": str(sb),
                "n": int(len(v)),
                "mean_fwd20": float(v.mean()) if len(v) else float("nan"),
                "median_fwd20": float(v.median()) if len(v) else float("nan"),
            }
        )
    by_cell = pd.DataFrame(rows).sort_values(["day", "joint_tight", "s_ext_bucket"])
    by_cell.to_csv(OUT / "r4_burst_m01_m22_fwd20_by_day_gate_extspread.csv", index=False)

    # Day-only pooled burst stats
    day_sum = (
        raw.groupby("day")["fwd20_ex"]
        .agg(n="count", mean_fwd20="mean", median_fwd20="median")
        .reset_index()
    )
    day_sum.to_csv(OUT / "r4_burst_m01_m22_fwd20_by_day_only.csv", index=False)

    summary = {
        "definition": "Burst: >=3 VEV_* trades same timestamp, buyer Mark 01, seller Mark 22. fwd20: m_ext(t+K)-m_ext(t) on price tape rows.",
        "n_burst_ticks_total": int(len(raw)),
        "by_day": day_sum.to_dict(orient="records"),
        "day3_mean_fwd20": float(raw.loc[raw["day"] == 3, "fwd20_ex"].mean()),
        "day12_mean_fwd20": float(raw.loc[raw["day"].isin([1, 2]), "fwd20_ex"].mean()),
        "joint_tight_fraction_at_burst": float(raw["joint_tight"].mean()),
    }
    (OUT / "r4_burst_m01_m22_day_stability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
