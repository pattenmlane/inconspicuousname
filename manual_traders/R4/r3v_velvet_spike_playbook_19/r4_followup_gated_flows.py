#!/usr/bin/env python3
"""Post-Phase-3: where Sonic gate binds (wide bursts), hydro pair under gate, Mark55 extract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH, K, DAYS = 2, 20, (1, 2, 3)


def load_prices(day: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")


def one_spread_mid(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = df[df["product"] == product].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(spread=ask - bid, mid=mid)[["timestamp", "spread", "mid"]]


def gate_panel(day: int) -> pd.DataFrame:
    df = load_prices(day)
    a = one_spread_mid(df, "VEV_5200").rename(columns={"spread": "s5200"})
    b = one_spread_mid(df, "VEV_5300").rename(columns={"spread": "s5300"})
    m = a.merge(b, on="timestamp")
    m["joint_tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["day"] = day
    return m[["day", "timestamp", "joint_tight", "s5200", "s5300"]]


def fwd_mid(df: pd.DataFrame, product: str, k: int) -> pd.Series:
    v = df[df["product"] == product].drop_duplicates("timestamp").sort_values("timestamp")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(fwd=mid.shift(-k) - mid).set_index("timestamp")["fwd"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gp = pd.concat([gate_panel(d) for d in DAYS], ignore_index=True)
    fwd_h = {}
    fwd_x = {}
    for d in DAYS:
        df = load_prices(d)
        fwd_h[d] = fwd_mid(df, "HYDROGEL_PACK", K)
        fwd_x[d] = fwd_mid(df, "VELVETFRUIT_EXTRACT", K)

    trades = []
    for d in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        trades.append(t)
    T = pd.concat(trades, ignore_index=True)
    T = T.merge(gp, on=["day", "timestamp"], how="inner")

    # Wide-gate M01-22 VEV burst
    T["n_tick"] = T.groupby(["day", "timestamp"])["symbol"].transform("count")
    mb = T[
        (T["n_tick"] >= 3)
        & (T["buyer"] == "Mark 01")
        & (T["seller"] == "Mark 22")
        & T["symbol"].astype(str).str.startswith("VEV_")
    ]
    wide = mb[~mb["joint_tight"]]
    summ_wide = {"n_rows": int(len(wide)), "by_day": wide.groupby("day").size().to_dict()}
    if len(wide):
        fx = wide.apply(
            lambda r: fwd_x[int(r["day"])].get(int(r["timestamp"]), np.nan), axis=1
        )
        summ_wide["mean_fwd20_extract"] = float(fx.mean())
    (OUT / "r4_followup_wide_gate_m01_m22_burst.json").write_text(json.dumps(summ_wide, indent=2))

    # Hydro 14-38 or 38-14
    hy = T[T["symbol"] == "HYDROGEL_PACK"].copy()
    pair = (
        ((hy["buyer"] == "Mark 14") & (hy["seller"] == "Mark 38"))
        | ((hy["buyer"] == "Mark 38") & (hy["seller"] == "Mark 14"))
    )
    hy = hy[pair]
    hy["fwd20_hydro"] = hy.apply(
        lambda r: fwd_h[int(r["day"])].get(int(r["timestamp"]), np.nan), axis=1
    )
    hsum = (
        hy.groupby(["day", "joint_tight"])["fwd20_hydro"]
        .agg(n="count", mean="mean")
        .reset_index()
    )
    hsum.to_csv(OUT / "r4_followup_hydro_14_38_fwd20_by_gate.csv", index=False)

    # Mark 55 on extract
    ex = T[T["symbol"] == "VELVETFRUIT_EXTRACT"].copy()
    ex = ex[(ex["buyer"] == "Mark 55") | (ex["seller"] == "Mark 55")]
    ex["fwd20_ex"] = ex.apply(
        lambda r: fwd_x[int(r["day"])].get(int(r["timestamp"]), np.nan), axis=1
    )
    m55 = (
        ex.groupby(["day", "joint_tight"])["fwd20_ex"]
        .agg(n="count", mean="mean", median="median")
        .reset_index()
    )
    m55.to_csv(OUT / "r4_followup_mark55_extract_fwd20_by_gate.csv", index=False)

    print("wrote followups to", OUT)


if __name__ == "__main__":
    main()
