#!/usr/bin/env python3
"""
Phase 1 burst structure — orchestrator attribution at multi-print timestamps.

For each (day, timestamp) with >1 trade row (any symbols), count how often each name appears
as buyer vs seller across prints at that instant; label dominant_buyer / dominant_seller
(break ties lexicographically for reproducibility). Join extract forward dm_ex_k20 from any
row at that key (identical across symbols for same t). Optionally merge sonic_tight from
r4_p3_trade_enriched_with_gate.csv when present.

Outputs (analysis_outputs/):
- r4_p1_burst_orchestrator_events.csv — one row per burst timestamp
- r4_p1_burst_pair_extract_k20.csv — groupby (dominant_buyer, dominant_seller): mean, count
- r4_p1_burst_dominant_buyer_extract_k20.csv — groupby dominant_buyer only
- r4_p1_burst_extract_k20_by_gate.csv — burst rows with sonic_tight vs not (if gate file exists)
- r4_p1_burst_orchestrator_summary.json — counts, top pairs, sonic overlap fraction
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "analysis_outputs")
ENRICHED = os.path.join(OUT, "r4_p1_trade_enriched.csv")
GATE = os.path.join(OUT, "r4_p3_trade_enriched_with_gate.csv")


def dominant(counter: Counter[str]) -> str:
    if not counter:
        return ""
    mx = max(counter.values())
    cands = sorted([k for k, v in counter.items() if v == mx and k])
    return cands[0] if cands else ""


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(ENRICHED)
    if "burst" not in df.columns:
        bc = df.groupby(["day", "timestamp"]).size().reset_index(name="n_prints")
        burst_set = set(zip(bc.loc[bc["n_prints"] > 1, "day"], bc.loc[bc["n_prints"] > 1, "timestamp"]))
        df["burst"] = [((int(a), int(b)) in burst_set) for a, b in zip(df["day"], df["timestamp"])]
    bdf = df[df["burst"]].copy()

    events: list[dict[str, Any]] = []
    for (day, ts), g in bdf.groupby(["day", "timestamp"]):
        buyers = Counter()
        sellers = Counter()
        for _, r in g.iterrows():
            b = str(r["buyer"]) if pd.notna(r["buyer"]) else ""
            s = str(r["seller"]) if pd.notna(r["seller"]) else ""
            if b:
                buyers[b] += 1
            if s:
                sellers[s] += 1
        db, ds = dominant(buyers), dominant(sellers)
        ex = g["dm_ex_k20"].dropna()
        exm = float(ex.iloc[0]) if len(ex) else float("nan")
        events.append(
            {
                "day": int(day),
                "timestamp": int(ts),
                "n_prints": int(len(g)),
                "n_symbols": int(g["symbol"].nunique()),
                "dominant_buyer": db,
                "dominant_seller": ds,
                "pair": f"{db}|{ds}",
                "dm_ex_k20": exm,
            }
        )

    ev = pd.DataFrame(events)
    ev.to_csv(os.path.join(OUT, "r4_p1_burst_orchestrator_events.csv"), index=False)

    pair = (
        ev.groupby("pair", dropna=False)["dm_ex_k20"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_dm_ex_k20", "count": "n_bursts"})
        .sort_values("n_bursts", ascending=False)
    )
    pair.to_csv(os.path.join(OUT, "r4_p1_burst_pair_extract_k20.csv"), index=False)

    buyonly = (
        ev.groupby("dominant_buyer", dropna=False)["dm_ex_k20"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_dm_ex_k20", "count": "n_bursts"})
        .sort_values("n_bursts", ascending=False)
    )
    buyonly.to_csv(os.path.join(OUT, "r4_p1_burst_dominant_buyer_extract_k20.csv"), index=False)

    summary: dict[str, Any] = {
        "n_burst_timestamps": int(len(ev)),
        "mean_n_prints": float(ev["n_prints"].mean()) if len(ev) else None,
        "pooled_mean_dm_ex_k20": float(ev["dm_ex_k20"].mean()) if len(ev) else None,
        "top_5_pairs_by_count": pair.head(5).to_dict(orient="records"),
    }

    if os.path.isfile(GATE):
        gdf = pd.read_csv(GATE)[["day", "timestamp", "sonic_tight"]].drop_duplicates()
        ev2 = ev.merge(gdf, on=["day", "timestamp"], how="left")
        ev2["sonic_tight"] = ev2["sonic_tight"].fillna(False)
        rows = []
        for tight, lab in [(True, "sonic_tight"), (False, "sonic_loose")]:
            sub = ev2[ev2["sonic_tight"] == tight]["dm_ex_k20"].dropna()
            rows.append(
                {
                    "slice": lab,
                    "n": int(len(sub)),
                    "mean_dm_ex_k20": float(sub.mean()) if len(sub) else float("nan"),
                }
            )
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "r4_p1_burst_extract_k20_by_gate.csv"), index=False)
        summary["sonic_tight_frac_of_burst_ts"] = float(ev2["sonic_tight"].mean()) if len(ev2) else None
        summary["burst_by_gate"] = rows
    else:
        summary["burst_by_gate"] = []

    m01 = ev[ev["pair"] == "Mark 01|Mark 22"].copy()
    day_rows = []
    for d, g in m01.groupby("day"):
        day_rows.append(
            {
                "day": int(d),
                "n_bursts": int(len(g)),
                "mean_dm_ex_k20": float(g["dm_ex_k20"].mean()) if len(g) else float("nan"),
            }
        )
    pd.DataFrame(day_rows).to_csv(os.path.join(OUT, "r4_p1_burst_m01_m22_dominant_by_day.csv"), index=False)
    summary["dominant_m01_m22_burst_by_day"] = day_rows

    if os.path.isfile(GATE):
        gdf = pd.read_csv(GATE)[["day", "timestamp", "sonic_tight"]].drop_duplicates()
        m01g = m01.merge(gdf, on=["day", "timestamp"], how="left")
        m01g["sonic_tight"] = m01g["sonic_tight"].fillna(False)
        gx = []
        for d, g in m01g.groupby("day"):
            for tight, lab in [(True, "tight"), (False, "loose")]:
                sub = g[g["sonic_tight"] == tight]["dm_ex_k20"].dropna()
                gx.append(
                    {
                        "day": int(d),
                        "gate": lab,
                        "n": int(len(sub)),
                        "mean_dm_ex_k20": float(sub.mean()) if len(sub) else float("nan"),
                    }
                )
        pd.DataFrame(gx).to_csv(os.path.join(OUT, "r4_p1_burst_m01_m22_dominant_by_day_gate.csv"), index=False)
        summary["dominant_m01_m22_burst_day_x_gate"] = gx

    with open(os.path.join(OUT, "r4_p1_burst_orchestrator_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
