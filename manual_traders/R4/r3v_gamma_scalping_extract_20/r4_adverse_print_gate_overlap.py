#!/usr/bin/env python3
"""
Round 4 — **adverse counterparty** (Mark 14 → Mark 55 on extract) overlap with
**Sonic joint tight** timestamps, and forward **VEV_5200** mid markout (K tape steps).

Motivation: Phase 1/2 flagged Mark14→Mark55 on extract at long horizon; Phase 3
showed day instability in tight-only tables. Here we quantify **how often**
that print co-occurs with joint tight and whether **VEV_5200** forward mid
differs after such ticks vs other tight ticks (same-day).

Outputs under analysis_outputs/:
  r4_adverse_tight_overlap_by_day.csv
  r4_adverse_vev5200_fwd_after_print.csv
"""
from __future__ import annotations

import bisect
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
K = 20
EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"


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


def load_tr() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            t = pd.read_csv(p, sep=";")
            t["day"] = d
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def tight_ts_set(px: pd.DataFrame, day: int) -> set[int]:
    def spread_df(prod: str) -> pd.DataFrame:
        v = (
            px[(px["day"] == day) & (px["product"] == prod)]
            .drop_duplicates(subset=["timestamp"], keep="first")
            .sort_values("timestamp")
        )
        bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
        return pd.DataFrame(
            {"timestamp": v["timestamp"].astype(int), "spread": (ask - bid).astype(float)}
        )

    a = spread_df("VEV_5200")
    b = spread_df("VEV_5300")
    m = a.merge(b, on="timestamp", suffixes=("_52", "_53"))
    m = m[(m["spread_52"] <= TH) & (m["spread_53"] <= TH)]
    return set(m["timestamp"].astype(int).tolist())


def mid_series(px: pd.DataFrame, day: int, prod: str) -> tuple[np.ndarray, np.ndarray]:
    v = px[(px["day"] == day) & (px["product"] == prod)].drop_duplicates("timestamp").sort_values("timestamp")
    return v["timestamp"].astype(int).values, pd.to_numeric(v["mid_price"], errors="coerce").values


def fwd_at(ts_arr: np.ndarray, mid_arr: np.ndarray, ts: int, k: int) -> float | None:
    i = bisect.bisect_left(ts_arr, ts)
    if i >= len(ts_arr) or ts_arr[i] != ts:
        return None
    j = i + k
    if j >= len(ts_arr):
        return None
    a, b = float(mid_arr[i]), float(mid_arr[j])
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return b - a


def main() -> int:
    px = load_px()
    tr = load_tr()

    overlap_rows = []
    fwd_rows = []

    for d in DAYS:
        tight = tight_ts_set(px, d)
        adv = set(
            tr.loc[
                (tr["day"] == d)
                & (tr["buyer"] == "Mark 14")
                & (tr["seller"] == "Mark 55")
                & (tr["symbol"] == EXTRACT),
                "timestamp",
            ]
            .astype(int)
        )
        both = tight & adv
        overlap_rows.append(
            {
                "day": d,
                "n_tight_ticks": len(tight),
                "n_adverse_print_ts": len(adv),
                "n_tight_and_adverse": len(both),
                "frac_tight_with_adverse": len(both) / max(1, len(tight)),
            }
        )

        ts52, m52 = mid_series(px, d, VEV_5200)
        for ts in sorted(tight):
            dm = fwd_at(ts52, m52, int(ts), K)
            if dm is None:
                continue
            fwd_rows.append(
                {
                    "day": d,
                    "timestamp": int(ts),
                    "adverse_tick": int(ts) in both,
                    "d_mid_vev5200_k20": dm,
                }
            )

    pd.DataFrame(overlap_rows).to_csv(OUT / "r4_adverse_tight_overlap_by_day.csv", index=False)

    df = pd.DataFrame(fwd_rows)
    if not df.empty:
        summ = (
            df.groupby(["day", "adverse_tick"])["d_mid_vev5200_k20"]
            .agg(n="count", mean="mean", std="std")
            .reset_index()
        )
        summ.to_csv(OUT / "r4_adverse_vev5200_fwd_after_print.csv", index=False)
        print(summ.to_string())
    print(pd.DataFrame(overlap_rows).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
