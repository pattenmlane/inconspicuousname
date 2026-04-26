#!/usr/bin/env python3
"""Mark 55 aggressive-buy EXTRACT × Sonic gate: Welch tight vs loose for fwd K ∈ {5,20,100}."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
P1 = Path(__file__).resolve().parent / "outputs_r4_phase1" / "r4_p1_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "outputs_r4_phase3" / "r4_p5_mark55_gate_multik_welch.csv"
DAYS = [1, 2, 3]
TH = 2


def _one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    v = v.assign(spread=(ask - bid).astype(float))
    return v[["timestamp", "spread"]].copy()


def gate_frame(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    a = _one_product(df, "VEV_5200").rename(columns={"spread": "s5200"})
    b = _one_product(df, "VEV_5300").rename(columns={"spread": "s5300"})
    m = a.merge(b, on="timestamp", how="inner")
    m["day"] = day
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m[["day", "timestamp", "tight"]]


def main() -> None:
    tr = pd.read_csv(P1)
    gates = pd.concat([gate_frame(d) for d in DAYS], ignore_index=True)
    mg = tr.merge(gates, on=["day", "timestamp"], how="left")
    mg["tight"] = mg["tight"].fillna(False)

    sub = mg[
        (mg["symbol"] == "VELVETFRUIT_EXTRACT")
        & (mg["buyer"] == "Mark 55")
        & (mg["aggressor_bucket"] == "aggr_buy")
    ]
    rows = []
    for k in (5, 20, 100):
        col = f"fwd_mid_k{k}"
        x = pd.to_numeric(sub[col], errors="coerce")
        a = x[sub["tight"]].dropna().to_numpy(dtype=float)
        b = x[~sub["tight"]].dropna().to_numpy(dtype=float)
        if len(a) > 1 and len(b) > 1:
            r = stats.ttest_ind(a, b, equal_var=False)
            tstat, pval = float(r.statistic), float(r.pvalue)
        else:
            tstat, pval = float("nan"), float("nan")
        rows.append(
            {
                "K": k,
                "n_tight": int(len(a)),
                "mean_tight": float(np.nanmean(a)) if len(a) else float("nan"),
                "n_loose": int(len(b)),
                "mean_loose": float(np.nanmean(b)) if len(b) else float("nan"),
                "welch_t": tstat,
                "welch_p": pval,
            }
        )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
