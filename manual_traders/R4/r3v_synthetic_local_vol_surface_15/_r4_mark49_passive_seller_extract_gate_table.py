#!/usr/bin/env python3
"""Mark 49 seller on aggr_buy EXTRACT: fwd20 by day × joint gate + pooled Welch row."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
P1 = BASE / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
PAN = BASE / "outputs_r4_phase3" / "r4_p3_joint_gate_panel_by_timestamp.csv"
OUT_DAY = BASE / "outputs_r4_phase3" / "r4_p18_mark49_seller_aggr_buy_extract_gate_by_day.csv"
OUT_WELCH = BASE / "outputs_r4_phase3" / "r4_p18_mark49_seller_aggr_buy_extract_gate_welch.csv"
MAN = BASE / "outputs_r4_phase3" / "r4_p3_manifest.json"


def main() -> None:
    tr = pd.read_csv(P1)
    pan = pd.read_csv(PAN, usecols=["day", "timestamp", "tight"]).drop_duplicates()
    mg = tr.merge(pan, on=["day", "timestamp"], how="left")
    mg["tight"] = mg["tight"].fillna(False)

    sub = mg[
        (mg["symbol"] == "VELVETFRUIT_EXTRACT")
        & (mg["seller"] == "Mark 49")
        & (mg["aggressor_bucket"] == "aggr_buy")
    ].copy()
    sub["fwd20"] = pd.to_numeric(sub["fwd_mid_k20"], errors="coerce")

    rows = []
    for d in sorted(sub["day"].unique()):
        s = sub[sub["day"] == d]
        for gate, lab in [(True, "tight"), (False, "loose")]:
            v = s.loc[s["tight"] == gate, "fwd20"].dropna()
            if len(v) < 1:
                continue
            rows.append(
                {
                    "day": int(d),
                    "gate": lab,
                    "n": int(len(v)),
                    "mean_fwd20": float(v.mean()),
                    "median_fwd20": float(v.median()),
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DAY, index=False)

    x = sub["fwd20"]
    t = x[sub["tight"]].dropna().to_numpy()
    f = x[~sub["tight"]].dropna().to_numpy()
    if len(t) > 1 and len(f) > 1:
        r = stats.ttest_ind(t, f, equal_var=False)
        wrow = pd.DataFrame(
            [
                {
                    "slice": "Mark49_passive_seller_EXTRACT_aggr_buy_fwd20",
                    "n_tight": int(len(t)),
                    "mean_tight": float(np.mean(t)),
                    "n_loose": int(len(f)),
                    "mean_loose": float(np.mean(f)),
                    "welch_t": float(r.statistic),
                    "welch_p": float(r.pvalue),
                }
            ]
        )
    else:
        wrow = pd.DataFrame()
    wrow.to_csv(OUT_WELCH, index=False)
    print(wrow.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))

    if MAN.is_file():
        m = json.loads(MAN.read_text(encoding="utf-8"))
        outs = set(m.get("outputs", []))
        outs.add(OUT_DAY.name)
        outs.add(OUT_WELCH.name)
        m["outputs"] = sorted(outs)
        MAN.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print("Wrote", OUT_DAY, OUT_WELCH)


if __name__ == "__main__":
    main()
