#!/usr/bin/env python3
"""
Bootstrap 95% CI for Phase-1 flagship VELVETFRUIT_EXTRACT aggr_buy K=5 fwd_same edges.

Reads precomputed events CSV from Phase-1 pipeline:
  outputs/phase1/events_with_cell_residual.csv

Outputs:
  outputs/phase1/extract_aggrbuy_k5_bootstrap_ci.csv

Run after r4_phase1_counterparty_analysis.py:
  python3 manual_traders/R4/r3v_jump_gap_filter_17/r4_phase1_bootstrap_extract_edges.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
EVENTS = OUT / "events_with_cell_residual.csv"
OUT_CSV = OUT / "extract_aggrbuy_k5_bootstrap_ci.csv"

B = 8000
RNG = np.random.default_rng(42)
K = 5
EX = "VELVETFRUIT_EXTRACT"


def boot_mean(x: np.ndarray) -> tuple[float, float, float]:
    n = len(x)
    if n < 2:
        return (float(np.nan), float(np.nan), float(np.nan))
    mu = float(np.mean(x))
    idx = RNG.integers(0, n, size=(B, n))
    samples = x[idx].mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (mu, float(lo), float(hi))


def main() -> None:
    ev = pd.read_csv(EVENTS)
    ev = ev[(ev["symbol"] == EX) & (ev["K"] == K) & (ev["aggressor"] == "aggr_buy")].copy()
    specs: list[tuple[str, str]] = [
        ("Mark 67 buyer", "buyer", "Mark 67"),
        ("Mark 22 seller", "seller", "Mark 22"),
        ("Mark 49 seller", "seller", "Mark 49"),
    ]
    rows = []
    for label, role, name in specs:
        if role == "buyer":
            sub = ev[ev["buyer"] == name]
        else:
            sub = ev[ev["seller"] == name]
        x = sub["fwd_same"].astype(float).values
        x = x[np.isfinite(x)]
        n = len(x)
        mu, lo, hi = boot_mean(x)
        rows.append(
            {
                "signal": label,
                "n": n,
                "mean_fwd_same": mu,
                "ci95_low": lo,
                "ci95_high": hi,
                "B": B,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print("Wrote", OUT_CSV)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
