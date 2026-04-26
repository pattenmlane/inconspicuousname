#!/usr/bin/env python3
"""
Phase 1 bullet 1 — **cross-asset** forward mids from participant prints.

Uses rows from participant_markout_long.csv (one row per print event × name tag × k):
  fwd_extract = VELVETFRUIT_EXTRACT mid(t+k) - mid(t) at trade timestamp t
  fwd_hydro   = HYDROGEL_PACK mid(t+k) - mid(t)

Aggregates mean / median / t-stat / frac_pos / n by stratification keys, **by tape_day**
and **pooled** across days. Writes tables for auditable Phase-1 completion.

Run: python3 manual_traders/R4/r3v_wing_vs_core_spread_04/r4_phase1_cross_asset_markouts.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
LONG = OUT / "participant_markout_long.csv"


def agg_stats(y: np.ndarray) -> dict:
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return {"n": n, "mean": float("nan"), "median": float("nan"), "t_stat": float("nan"), "frac_pos": float("nan")}
    mean = float(y.mean())
    std = float(y.std(ddof=1))
    t_stat = mean / (std / math.sqrt(n)) if std > 1e-12 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(y)),
        "t_stat": t_stat,
        "frac_pos": float((y > 0).mean()),
    }


def main() -> None:
    if not LONG.is_file():
        raise SystemExit(f"Missing {LONG}; run r4_phase1_counterparty_analysis.py first")

    df = pd.read_csv(LONG)
    for col in ("fwd_extract", "fwd_hydro"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows_long: list[dict] = []
    for target, col in (("VELVETFRUIT_EXTRACT", "fwd_extract"), ("HYDROGEL_PACK", "fwd_hydro")):
        sub = df[["tape_day", "name", "role", "symbol", "k", "spread_regime", "burst", col]].copy()
        sub = sub.rename(columns={col: "fwd_cross"})
        sub["cross_target"] = target
        rows_long.append(sub)
    x = pd.concat(rows_long, ignore_index=True)

    gcols = ["cross_target", "name", "role", "symbol", "k", "spread_regime", "burst", "tape_day"]
    by_day: list[dict] = []
    for key, g in x.groupby(gcols, dropna=False):
        st = agg_stats(g["fwd_cross"].to_numpy(dtype=float))
        if st["n"] == 0:
            continue
        d = dict(zip(gcols, key, strict=True))
        d.update(st)
        by_day.append(d)
    by_day_df = pd.DataFrame(by_day)

    gcols_p = ["cross_target", "name", "role", "symbol", "k", "spread_regime", "burst"]
    pooled: list[dict] = []
    for key, g in x.groupby(gcols_p, dropna=False):
        st = agg_stats(g["fwd_cross"].to_numpy(dtype=float))
        d = dict(zip(gcols_p, key, strict=True))
        d["n_days_nonzero"] = int(g["tape_day"].nunique())
        d.update(st)
        pooled.append(d)
    pool_df = pd.DataFrame(pooled)

    by_day_df.to_csv(OUT / "participant_cross_asset_markout_by_day.csv", index=False)
    pool_df.to_csv(OUT / "participant_cross_asset_markout_pooled.csv", index=False)

    # Headline: Mark 67 buy_aggr on extract -> fwd_extract and fwd_hydro, tight, burst=0, k=20
    def headline(pool: pd.DataFrame, cross: str) -> list[dict]:
        m = (
            (pool["cross_target"] == cross)
            & (pool["name"] == "Mark 67")
            & (pool["role"] == "buy_aggr")
            & (pool["symbol"] == "VELVETFRUIT_EXTRACT")
            & (pool["k"] == 20)
            & (pool["spread_regime"] == "tight")
            & (pool["burst"] == 0)
        )
        hit = pool.loc[m]
        if len(hit) == 0:
            return []
        row = hit.iloc[0].to_dict()
        days = by_day_df[
            (by_day_df["cross_target"] == cross)
            & (by_day_df["name"] == "Mark 67")
            & (by_day_df["role"] == "buy_aggr")
            & (by_day_df["symbol"] == "VELVETFRUIT_EXTRACT")
            & (by_day_df["k"] == 20)
            & (by_day_df["spread_regime"] == "tight")
            & (by_day_df["burst"] == 0)
        ][["tape_day", "n", "mean", "t_stat", "frac_pos"]]
        return [
            {
                "cross_target": cross,
                "pooled": row,
                "per_day": days.sort_values("tape_day").to_dict(orient="records"),
            }
        ]

    hl = headline(pool_df, "VELVETFRUIT_EXTRACT") + headline(pool_df, "HYDROGEL_PACK")
    (OUT / "cross_asset_m67_buy_aggr_extract_tight_k20_headline.json").write_text(
        json.dumps(hl, indent=2), encoding="utf-8"
    )

    # Top pooled positive mean on fwd_extract (n>=30, k=20, tight, burst=0)
    top_ex = pool_df[
        (pool_df["cross_target"] == "VELVETFRUIT_EXTRACT")
        & (pool_df["k"] == 20)
        & (pool_df["spread_regime"] == "tight")
        & (pool_df["burst"] == 0)
        & (pool_df["n"] >= 30)
    ].sort_values("mean", ascending=False)
    top_ex.head(25).to_csv(OUT / "cross_asset_fwd_extract_k20_tight_top25_pooled.csv", index=False)

    top_h = pool_df[
        (pool_df["cross_target"] == "HYDROGEL_PACK")
        & (pool_df["k"] == 20)
        & (pool_df["spread_regime"] == "tight")
        & (pool_df["burst"] == 0)
        & (pool_df["n"] >= 30)
    ].sort_values("mean", ascending=False)
    top_h.head(25).to_csv(OUT / "cross_asset_fwd_hydro_k20_tight_top25_pooled.csv", index=False)

    print("Wrote cross-asset tables to", OUT)


if __name__ == "__main__":
    main()
