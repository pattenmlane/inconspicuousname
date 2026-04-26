#!/usr/bin/env python3
"""
Round 4 Phase 1 — stratifications requested in suggested direction:
- **Hour bucket** (already in CSV as hour_b): pooled mean fwd + t-stat for aggressor prints.
- **Burst vs isolated** at (day, timestamp): n_trades at that timestamp across *all* products
  — burst if n>=3, isolated if n==1, else "pair" (n==2).

Outputs (under analysis_outputs/):
- r4_phase1_aggressor_fwd_by_hour_product_pooled.csv
- r4_phase1_aggressor_fwd_by_ts_density_pooled.csv
- r4_phase1_burst_hour_stratify_index.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INP = Path(__file__).resolve().parent / "analysis_outputs" / "r4_trades_with_markout.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
KS = (5, 20, 100)
MIN_N_HOUR = 30
MIN_N_TS = 10


def t_one(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(x, 0.0, nan_policy="omit")
    return float(t), float(p)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(INP)
    ts_n = m.groupby(["day", "timestamp"]).size().rename("n_at_ts").reset_index()
    m = m.merge(ts_n, on=["day", "timestamp"], how="left")
    m["ts_density"] = np.where(
        m["n_at_ts"] >= 3, "burst_ge3", np.where(m["n_at_ts"] == 1, "isolated", "pair_n2")
    )

    m_ag = m[m["aggr"].isin(["buy_aggr", "sell_aggr"])].copy()

    # --- By hour_b x product (same-symbol fwd), aggressor only ---
    rows_h = []
    for k in KS:
        col = f"fwd_{k}_sym"
        for (hb, prod), g in m_ag.groupby(["hour_b", "product"]):
            x = g[col].dropna().to_numpy(dtype=float)
            n = len(x)
            if n < MIN_N_HOUR:
                continue
            tm, pval = t_one(x)
            rows_h.append(
                {
                    "K": k,
                    "hour_b": str(hb),
                    "product": str(prod),
                    "n": n,
                    "mean": float(x.mean()),
                    "median": float(np.median(x)),
                    "frac_pos": float((x > 0).mean()),
                    "tstat_vs0": tm,
                    "pvalue_vs0": pval,
                }
            )
    pd.DataFrame(rows_h).to_csv(OUT / "r4_phase1_aggressor_fwd_by_hour_product_pooled.csv", index=False)

    # --- By ts_density x product ---
    rows_t = []
    for k in KS:
        col = f"fwd_{k}_sym"
        for (td, prod), g in m_ag.groupby(["ts_density", "product"]):
            x = g[col].dropna().to_numpy(dtype=float)
            n = len(x)
            if n < MIN_N_TS:
                continue
            tm, pval = t_one(x)
            rows_t.append(
                {
                    "K": k,
                    "ts_density": str(td),
                    "product": str(prod),
                    "n": n,
                    "mean": float(x.mean()),
                    "median": float(np.median(x)),
                    "frac_pos": float((x > 0).mean()),
                    "tstat_vs0": tm,
                    "pvalue_vs0": pval,
                }
            )
    pd.DataFrame(rows_t).to_csv(OUT / "r4_phase1_aggressor_fwd_by_ts_density_pooled.csv", index=False)

    # Cross-asset: extract forward at every print (not only aggressor) by ts_density — use all rows on extract product
    rows_x = []
    ex = m[m["product"] == "VELVETFRUIT_EXTRACT"]
    for k in KS:
        col = f"fwd_{k}_ex"
        for td, g in ex.groupby("ts_density"):
            x = g[col].dropna().to_numpy(dtype=float)
            n = len(x)
            if n < MIN_N_TS:
                continue
            tm, pval = t_one(x)
            rows_x.append(
                {
                    "K": k,
                    "ts_density": str(td),
                    "forward_on": "VELVETFRUIT_EXTRACT_mid",
                    "n": n,
                    "mean": float(x.mean()),
                    "median": float(np.median(x)),
                    "tstat_vs0": tm,
                    "pvalue_vs0": pval,
                }
            )
    pd.DataFrame(rows_x).to_csv(OUT / "r4_phase1_extract_fwd_ex_by_ts_density_pooled.csv", index=False)

    idx = {
        "inputs": str(INP),
        "definition": {
            "burst_ge3": "At least 3 trade rows same (day,timestamp) any product.",
            "isolated": "Exactly 1 trade row at (day,timestamp).",
            "pair_n2": "Exactly 2 trade rows (excluded from burst vs isolated headline).",
            "horizon_K": "Forward rows in price tape for that product (same as Phase 1).",
        },
        "min_n": {"hour_x_product": MIN_N_HOUR, "ts_density": MIN_N_TS},
        "outputs": [
            str(OUT / "r4_phase1_aggressor_fwd_by_hour_product_pooled.csv"),
            str(OUT / "r4_phase1_aggressor_fwd_by_ts_density_pooled.csv"),
            str(OUT / "r4_phase1_extract_fwd_ex_by_ts_density_pooled.csv"),
        ],
    }
    (OUT / "r4_phase1_burst_hour_stratify_index.json").write_text(json.dumps(idx, indent=2), encoding="utf-8")
    print("Wrote hour + ts_density tables to", OUT)


if __name__ == "__main__":
    main()
