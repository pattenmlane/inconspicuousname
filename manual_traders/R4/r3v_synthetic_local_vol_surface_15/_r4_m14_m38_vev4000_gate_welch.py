#!/usr/bin/env python3
"""Welch t-test: Mark14 buy / Mark38 sell aggr_sell VEV_4000, fwd_mid_k20 tight vs loose (Phase-1 + gate)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
P1 = BASE / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
PAN = BASE / "outputs_r4_phase3" / "r4_p3_joint_gate_panel_by_timestamp.csv"
OUT = BASE / "outputs_r4_phase3" / "r4_p16_m14_m38_vev4000_gate_welch_fwd20.csv"
MAN = BASE / "outputs_r4_phase3" / "r4_p3_manifest.json"


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    r = stats.ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


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
    ]
    x = pd.to_numeric(sub["fwd_mid_k20"], errors="coerce")
    t = x[sub["tight"]].dropna().to_numpy()
    f = x[~sub["tight"]].dropna().to_numpy()
    ts, pv = welch(t, f)

    rows = [
        {
            "slice": "M14_buy_M38_sell_VEV4000_aggr_sell_fwd20_all_days",
            "n_tight": int(len(t)),
            "mean_tight": float(np.mean(t)) if len(t) else float("nan"),
            "n_loose": int(len(f)),
            "mean_loose": float(np.mean(f)) if len(f) else float("nan"),
            "welch_t": ts,
            "welch_p": pv,
        }
    ]
    for d in [1, 2, 3]:
        s = sub[sub["day"] == d]
        xx = pd.to_numeric(s["fwd_mid_k20"], errors="coerce")
        tt = xx[s["tight"]].dropna().to_numpy()
        ff = xx[~s["tight"]].dropna().to_numpy()
        t2, p2 = welch(tt, ff)
        rows.append(
            {
                "slice": f"day_{d}",
                "n_tight": int(len(tt)),
                "mean_tight": float(np.mean(tt)) if len(tt) else float("nan"),
                "n_loose": int(len(ff)),
                "mean_loose": float(np.mean(ff)) if len(ff) else float("nan"),
                "welch_t": t2,
                "welch_p": p2,
            }
        )

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    if MAN.is_file():
        m = json.loads(MAN.read_text(encoding="utf-8"))
        outs = set(m.get("outputs", []))
        outs.add(OUT.name)
        m["outputs"] = sorted(outs)
        MAN.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
