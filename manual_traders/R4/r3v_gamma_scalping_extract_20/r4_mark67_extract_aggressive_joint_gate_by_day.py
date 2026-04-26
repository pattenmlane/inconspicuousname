#!/usr/bin/env python3
"""
Mark 67 **aggressive extract buys** (tape trade price ≥ L1 ask) vs **Sonic joint
tight** (5200+5300 spread≤2) at the same timestamp — by tape day.

Outputs: analysis_outputs/r4_mark67_extract_agg_joint_gate_by_day.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = (1, 2, 3)
EXTRACT = "VELVETFRUIT_EXTRACT"
SURFACE = ("VEV_5200", "VEV_5300")
SPREAD_TH = 2
MARK67 = "Mark 67"


def joint_tight_frame(day: int) -> pd.DataFrame:
    px = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    sub = px[px["product"].isin(SURFACE)][["timestamp", "product", "bid_price_1", "ask_price_1"]].copy()
    sub["spread"] = pd.to_numeric(sub["ask_price_1"], errors="coerce") - pd.to_numeric(
        sub["bid_price_1"], errors="coerce"
    )
    pvt = sub.pivot_table(index="timestamp", columns="product", values="spread", aggfunc="first")
    pvt = pvt.dropna(subset=list(SURFACE))
    jt = (pvt["VEV_5200"] <= SPREAD_TH) & (pvt["VEV_5300"] <= SPREAD_TH)
    return pd.DataFrame({"timestamp": pvt.index.astype(int), "joint_tight": jt.values})


def extract_bbo(day: int) -> pd.DataFrame:
    px = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    ex = px[px["product"] == EXTRACT][["timestamp", "bid_price_1", "ask_price_1"]].copy()
    ex["timestamp"] = ex["timestamp"].astype(int)
    ex["bid"] = pd.to_numeric(ex["bid_price_1"], errors="coerce")
    ex["ask"] = pd.to_numeric(ex["ask_price_1"], errors="coerce")
    return ex[["timestamp", "bid", "ask"]]


def main() -> None:
    rows = []
    for d in DAYS:
        jt = joint_tight_frame(d)
        bbo = extract_bbo(d)
        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr = tr[tr["symbol"] == EXTRACT].copy()
        tr["timestamp"] = tr["timestamp"].astype(int)
        tr["buyer"] = tr["buyer"].fillna("").astype(str)
        tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
        tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
        m67 = tr[tr["buyer"] == MARK67]
        m = m67.merge(bbo, on="timestamp", how="left")
        m = m.merge(jt, on="timestamp", how="left")
        m["joint_tight"] = m["joint_tight"].fillna(False)
        m["aggressive"] = m["price"] >= m["ask"]
        n_m67 = len(m)
        n_agg = int(m["aggressive"].sum())
        n_agg_tight = int((m["aggressive"] & m["joint_tight"]).sum())
        share_tight_given_agg = (n_agg_tight / n_agg) if n_agg else 0.0
        rows.append(
            {
                "tape_day": d,
                "n_mark67_extract_trades": n_m67,
                "n_aggressive_buy_price_ge_ask": n_agg,
                "n_aggressive_and_joint_tight": n_agg_tight,
                "share_joint_tight_given_aggressive": round(share_tight_given_agg, 6),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_mark67_extract_agg_joint_gate_by_day.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
