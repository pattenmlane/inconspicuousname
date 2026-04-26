#!/usr/bin/env python3
"""
Phase 1 bullet 4 supplement: burst orchestrator attribution vs forward extract markout.

Burst: >=4 trade rows same (day, timestamp).
Orchestrator buyer = mode(buyer) within burst; orchestrator seller = mode(seller).
Forward U@20: take mark_20_u from enriched trades (same for all rows at a timestamp) — one row per (day,timestamp).

Outputs:
  r4_phase1_burst_orchestrator_forward_u.csv  (one row per burst event)
  r4_phase1_burst_orch_buyer_summary.csv      (pooled mean U@20 by orch buyer)
  r4_phase1_burst_orch_pair_summary.csv       (top (orch_buyer, orch_seller) cells, min n bursts)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
ENR = Path(__file__).resolve().parent / "outputs" / "r4_trades_enriched_markouts.csv"
OUT = Path(__file__).resolve().parent / "outputs"
MIN_SUMMARY = 8


def mode_nonempty(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    s = s[s != ""]
    if len(s) == 0:
        return ""
    return s.mode().iloc[0] if len(s.mode()) else str(s.iloc[0])


def main() -> None:
    parts = []
    for d in (1, 2, 3):
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        parts.append(t)
    tr = pd.concat(parts, ignore_index=True)

    g = tr.groupby(["day", "timestamp"])
    sz = g.size().rename("n_prints").reset_index()
    burst_ts = sz[sz["n_prints"] >= 4][["day", "timestamp"]]
    btr = tr.merge(burst_ts, on=["day", "timestamp"], how="inner")

    orch = (
        btr.groupby(["day", "timestamp"])
        .apply(
            lambda x: pd.Series(
                {
                    "n_prints": len(x),
                    "n_syms": x["symbol"].nunique(),
                    "orch_buyer": mode_nonempty(x["buyer"]),
                    "orch_seller": mode_nonempty(x["seller"]),
                }
            )
        )
        .reset_index()
    )

    enr = pd.read_csv(ENR, usecols=["day", "timestamp", "mark_20_u"])
    enr["mark_20_u"] = pd.to_numeric(enr["mark_20_u"], errors="coerce")
    uone = enr.groupby(["day", "timestamp"], as_index=False)["mark_20_u"].first()

    ev = orch.merge(uone, on=["day", "timestamp"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT / "r4_phase1_burst_orchestrator_forward_u.csv", index=False)

    summ_b = (
        ev.groupby("orch_buyer")["mark_20_u"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"count": "n_bursts", "mean": "mean_u20", "median": "median_u20"})
        .sort_values("n_bursts", ascending=False)
    )
    summ_b.to_csv(OUT / "r4_phase1_burst_orch_buyer_summary.csv", index=False)

    summ_p = (
        ev.groupby(["orch_buyer", "orch_seller"])["mark_20_u"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "n_bursts", "mean": "mean_u20"})
    )
    summ_p = summ_p[summ_p["n_bursts"] >= MIN_SUMMARY].sort_values("n_bursts", ascending=False)
    summ_p.to_csv(OUT / "r4_phase1_burst_orch_pair_summary.csv", index=False)
    print("wrote burst orchestrator tables, n_events", len(ev))


if __name__ == "__main__":
    main()
