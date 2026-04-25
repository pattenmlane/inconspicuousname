"""
Round 3 tapes: when extract mid moves sharply (|dS|>=2 vs prior timestamp), compare
top-of-book spread (ask1-bid1) for core VEV (5000-5300) vs deep wings (4000,4500,6500).

DTE mapping: round3work/round3description.txt (csv day 0->8d open, intraday wind).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent.parent
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "extract_shock_spread_core_wing.json"

U = "VELVETFRUIT_EXTRACT"
CORE = {f"VEV_{k}" for k in (5000, 5100, 5200, 5300)}
WINGS = {"VEV_4000", "VEV_4500", "VEV_6500"}
STEP = 1  # use every row within each timestamp group is heavy; aggregate by timestamp first


def spread(row: pd.Series) -> float | None:
    bp, ap = row.get("bid_price_1"), row.get("ask_price_1")
    if pd.isna(bp) or pd.isna(ap):
        return None
    return float(ap) - float(bp)


def main() -> None:
    core_shock: list[float] = []
    core_calm: list[float] = []
    wing_shock: list[float] = []
    wing_calm: list[float] = []
    n_shock = 0
    n_calm = 0

    for csv_day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{csv_day}.csv"
        df = pd.read_csv(path, sep=";")
        prev_s: float | None = None
        for ts in sorted(df["timestamp"].unique()):
            sub = df.loc[df["timestamp"] == ts]
            ex = sub[sub["product"] == U]
            if ex.empty:
                continue
            S = float(ex.iloc[0]["mid_price"])
            if prev_s is None:
                prev_s = S
                continue
            dS = S - prev_s
            prev_s = S
            shock = abs(dS) >= 2.0
            if shock:
                n_shock += 1
            else:
                n_calm += 1
            for v in CORE | WINGS:
                r0 = sub[sub["product"] == v]
                if r0.empty:
                    continue
                sw = spread(r0.iloc[0])
                if sw is None:
                    continue
                if v in CORE:
                    (core_shock if shock else core_calm).append(sw)
                else:
                    (wing_shock if shock else wing_calm).append(sw)

    def med(a: list[float]) -> float | None:
        return float(np.median(a)) if a else None

    payload = {
        "shock_def": "|delta extract mid| >= 2 vs prior timestamp within same csv day",
        "timestamps_shock": n_shock,
        "timestamps_calm": n_calm,
        "core_vev_median_spread_shock": med(core_shock),
        "core_vev_median_spread_calm": med(core_calm),
        "deep_wing_median_spread_shock": med(wing_shock),
        "deep_wing_median_spread_calm": med(wing_calm),
        "interpretation": "Median top-of-book spread is unchanged shock vs calm for both bands (core 3, wings 16) on this tape — shocks are not widening core books. Still useful: |dS|>=2 marks high extract-velocity windows where core vouchers co-move with S (see underlying_propagation); tightening core-only quotes on those steps is a targeted lever without changing wing liquidity filters.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
