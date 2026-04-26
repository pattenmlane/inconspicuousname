#!/usr/bin/env python3
"""Mark 14 buy / Mark 38 sell, aggr_sell on VEV_4000: fwd20 by day × joint gate (Phase-1 + r4_p3 panel)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
P1 = BASE / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
PAN = BASE / "outputs_r4_phase3" / "r4_p3_joint_gate_panel_by_timestamp.csv"
OUT = BASE / "outputs_r4_phase3" / "r4_p15_mark38_vev4000_m14_aggr_sell_by_gate_day.csv"
DAYS = [1, 2, 3]


def main() -> None:
    tr = pd.read_csv(P1)
    pan = pd.read_csv(PAN, usecols=["day", "timestamp", "tight"]).drop_duplicates()
    mg = tr.merge(pan, on=["day", "timestamp"], how="left")
    mg["tight"] = mg["tight"].fillna(False)

    sub = mg[
        (mg["symbol"] == "VEV_4000")
        & (mg["buyer"] == "Mark 14")
        & (mg["seller"] == "Mark 38")
        & (mg["aggressor_bucket"] == "aggr_sell")
    ].copy()
    sub["fwd20"] = pd.to_numeric(sub["fwd_mid_k20"], errors="coerce")

    rows = []
    for d in DAYS:
        for gate in [True, False]:
            lab = "tight" if gate else "loose"
            s = sub[(sub["day"] == d) & (sub["tight"] == gate)]["fwd20"].dropna()
            if len(s) < 1:
                continue
            rows.append(
                {
                    "day": d,
                    "gate": lab,
                    "n": int(len(s)),
                    "mean_fwd20": float(s.mean()),
                    "median_fwd20": float(s.median()),
                }
            )
    # pooled tight vs loose
    for gate, lab in [(True, "tight"), (False, "loose")]:
        s = sub[sub["tight"] == gate]["fwd20"].dropna()
        rows.append({"day": 0, "gate": lab + "_all_days", "n": int(len(s)), "mean_fwd20": float(s.mean()), "median_fwd20": float(s.median())})

    pd.DataFrame(rows).sort_values(["day", "gate"]).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
