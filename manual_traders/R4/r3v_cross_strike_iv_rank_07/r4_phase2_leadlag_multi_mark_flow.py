#!/usr/bin/env python3
"""
Round 4 Phase 2 — cross-instrument lead-lag: signed notional on VELVETFRUIT_EXTRACT per (day,timestamp)
for several Marks (buyer/seller × buy_agg/sell_agg) vs forward one-row extract mid change.

Reuses the same price-grid convention as r4_phase2_analysis.py: lag 0 correlation is n/a (or nan).
Output: r4_p2_leadlag_signed_flow_by_mark.csv — long form (mark, lag_price_rows, n, corr).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)
OUT = os.path.join(BASE, "analysis_outputs")
ENRICHED = os.path.join(OUT, "r4_p1_trade_enriched.csv")
PRICE_GLOB = "Prosperity4Data/ROUND_4/prices_round_4_day_{d}.csv"
DAYS = (1, 2, 3)
MARKS = [
    "Mark 67",
    "Mark 22",
    "Mark 01",
    "Mark 55",
    "Mark 14",
    "Mark 38",
    "Mark 49",
]
LAG_LIST = (1, 2, 3, 5, 10, 20)


def load_ex_mid_grid() -> pd.DataFrame:
    parts = []
    for d in DAYS:
        p = pd.read_csv(PRICE_GLOB.format(d=d), sep=";")
        p = p[p["product"] == "VELVETFRUIT_EXTRACT"][["timestamp", "mid_price"]].copy()
        p["day"] = d
        parts.append(p)
    ex = pd.concat(parts, ignore_index=True)
    return ex.sort_values(["day", "timestamp"])


def signed_flow_for_mark(df: pd.DataFrame, mark: str) -> pd.DataFrame:
    """Per (day, timestamp) sum signed notional for extract trades."""
    ex = df[df["symbol"] == "VELVETFRUIT_EXTRACT"].copy()
    ex["signed"] = 0.0
    m = (ex["buyer"] == mark) & (ex["agg"] == "buy_agg")
    ex.loc[m, "signed"] = ex.loc[m, "notional"]
    m2 = (ex["seller"] == mark) & (ex["agg"] == "sell_agg")
    ex.loc[m2, "signed"] = -ex.loc[m2, "notional"]
    g = ex.groupby(["day", "timestamp"], as_index=False)["signed"].sum()
    g = g.rename(columns={"signed": f"flow_{mark}"})
    return g


def main() -> None:
    if not os.path.isfile(ENRICHED):
        raise SystemExit(f"Missing {ENRICHED}; run r4_phase1_counterparty_analysis.py first")
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(ENRICHED)
    ex_px = load_ex_mid_grid()
    ex_px = ex_px.rename(columns={"mid_price": "mid_ex"})

    rows: list[dict] = []
    for mark in MARKS:
        flow = signed_flow_for_mark(df, mark)
        for lag in LAG_LIST:
            chunks = []
            for d in DAYS:
                g = ex_px[ex_px["day"] == d].reset_index(drop=True)
                if len(g) < lag + 100:
                    continue
                g["fwd_mid"] = g["mid_ex"].shift(-lag) - g["mid_ex"]
                mg = g.merge(flow[flow["day"] == d], on=["day", "timestamp"], how="left")
                cname = f"flow_{mark}"
                if cname not in mg.columns:
                    continue
                mg[cname] = mg[cname].fillna(0.0)
                sub = mg.dropna(subset=["fwd_mid"])
                if len(sub) > 200:
                    chunks.append(sub[[cname, "fwd_mid"]])
            if not chunks:
                continue
            z = pd.concat(chunks, ignore_index=True)
            fstd = float(z[cname].std(ddof=1)) if len(z) > 1 else 0.0
            if fstd < 1e-12:
                c = float("nan")
            else:
                c = float(z[cname].corr(z["fwd_mid"]))
            rows.append(
                {
                    "mark": mark,
                    "lag_price_rows": int(lag),
                    "lag_ts_units": int(lag * 100),
                    "n": int(len(z)),
                    "flow_stdev_pooled": fstd,
                    "corr": c,
                }
            )

    out = os.path.join(OUT, "r4_p2_leadlag_signed_flow_by_mark.csv")
    pd.DataFrame(rows).sort_values(["mark", "lag_price_rows"]).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
