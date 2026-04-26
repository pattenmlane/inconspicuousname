#!/usr/bin/env python3
"""
Phase 1 bullet 3 — **lead/lag** beyond two-hop counts: Pearson correlation of
tape-aligned **signed aggressive flow** (per symbol, per timestamp) vs **forward mid**
on the same price grid for other symbols.

- Flow: sum over trades at (tape_day, timestamp, symbol) of +qty (buy_aggr) / -qty (sell_aggr); 0 if none.
- Forward mid at grid index i: mid[i + K] - mid[i] for K in {5, 20} (tape steps).
- Corr(flow.shift(lag), fwd_target) for lag in {0,1,2,3,5} tape steps (flow **leads** when lag>0).

Outputs: outputs/phase1/cross_symbol_flow_fwd_mid_lag_corr.csv
          outputs/phase1/cross_symbol_flow_fwd_mid_lag_corr_pooled.csv (mean corr by cell across days)

Run: python3 manual_traders/R4/r3v_wing_vs_core_spread_04/r4_phase1_lead_lag_flow.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]
FLOW_SYMS = list(PRODUCTS)
TARGET_SYMS = ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK", "VEV_5200", "VEV_5300"]
K_FWD = (5, 20)
FLOW_LAGS = (0, 1, 2, 3, 5)


def load_prices(day: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            x = pd.read_csv(p, sep=";")
            x["tape_day"] = d
            frames.append(x)
    return pd.concat(frames, ignore_index=True)


def mid_panel(px: pd.DataFrame) -> pd.DataFrame:
    m = px.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="last")
    for p in PRODUCTS:
        if p not in m.columns:
            m[p] = np.nan
    return m[PRODUCTS]


def classify_aggression(row: pd.Series) -> str:
    try:
        p, b, a = float(row["price"]), float(row["bid_price_1"]), float(row["ask_price_1"])
    except (TypeError, ValueError, KeyError):
        return "unknown"
    if p >= a:
        return "buy_aggr"
    if p <= b:
        return "sell_aggr"
    return "inside"


def main() -> None:
    book_parts = []
    for d in DAYS:
        px = load_prices(d)
        b = px[px["product"].isin(PRODUCTS)].copy()
        b["tape_day"] = d
        book_parts.append(b)
    book = pd.concat(book_parts, ignore_index=True).rename(columns={"product": "symbol"})

    tr = load_trades()
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0)
    m = tr.merge(book, on=["tape_day", "timestamp", "symbol"], how="left")
    m["aggression"] = m.apply(classify_aggression, axis=1)
    m["signed_qty"] = np.where(
        m["aggression"] == "buy_aggr",
        m["quantity"],
        np.where(m["aggression"] == "sell_aggr", -m["quantity"], 0.0),
    )

    rows: list[dict] = []
    for d in DAYS:
        px = load_prices(d)
        mp = mid_panel(px)
        idx = mp.index.astype(int)
        T = len(mp)
        for tgt in TARGET_SYMS:
            if tgt not in mp.columns:
                continue
            mid = mp[tgt].astype(float)
            for k in K_FWD:
                fwd = mid.shift(-k) - mid
                fwd = fwd.reindex(idx).astype(float)
                for fs in FLOW_SYMS:
                    flow = (
                        m[(m["tape_day"] == d) & (m["symbol"] == fs)]
                        .groupby("timestamp")["signed_qty"]
                        .sum()
                        .reindex(idx)
                        .fillna(0.0)
                        .astype(float)
                    )
                    for lag in FLOW_LAGS:
                        x = flow.shift(lag)
                        pair = pd.concat([x, fwd], axis=1).dropna()
                        if len(pair) < 50:
                            c = float("nan")
                        else:
                            s0, s1 = float(pair.iloc[:, 0].std()), float(pair.iloc[:, 1].std())
                            if s0 < 1e-12 or s1 < 1e-12:
                                c = float("nan")
                            else:
                                cc = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                                c = float(cc) if not pd.isna(cc) else float("nan")
                        rows.append(
                            {
                                "tape_day": d,
                                "flow_symbol": fs,
                                "target_symbol": tgt,
                                "fwd_k": k,
                                "flow_lag_steps": lag,
                                "n": int(len(pair)),
                                "pearson_corr": c,
                            }
                        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cross_symbol_flow_fwd_mid_lag_corr.csv", index=False)
    pool = (
        df.groupby(["flow_symbol", "target_symbol", "fwd_k", "flow_lag_steps"], dropna=False)
        .agg(mean_corr=("pearson_corr", "mean"), n_days=("pearson_corr", "count"))
        .reset_index()
    )
    pool = pool.sort_values("mean_corr", key=lambda s: s.abs(), ascending=False)
    pool.to_csv(OUT / "cross_symbol_flow_fwd_mid_lag_corr_pooled.csv", index=False)

    # Top |corr| per day for quick read
    tops = []
    for d in DAYS:
        sub = df[df["tape_day"] == d].copy()
        sub["abs_c"] = sub["pearson_corr"].abs()
        tops.append(sub.sort_values("abs_c", ascending=False).head(12).drop(columns=["abs_c"]))
    pd.concat(tops, ignore_index=True).to_csv(OUT / "cross_symbol_flow_fwd_mid_lag_corr_top12_by_day.csv", index=False)
    print("Wrote", OUT / "cross_symbol_flow_fwd_mid_lag_corr.csv", "rows", len(df))


if __name__ == "__main__":
    main()
