#!/usr/bin/env python3
"""
Round 4 Phase 1 supplement — pooled (all tape days) aggressor cells with:
mean, median, n, one-sample t-statistic vs 0, bootstrap 95% CI for mean (percentile).

Reads r4_trades_with_markout.csv from r4_phase1_counterparty_study.py (must exist).
Horizon K ∈ {5,20,100} on same-symbol forward mid (fwd_{K}_sym).
Filters: aggressive only; min n >= 30 per cell (same as participant table).

Output: r4_phase1_aggressor_cell_stats_pooled.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
INP = Path(__file__).resolve().parent / "analysis_outputs" / "r4_trades_with_markout.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
KS = (5, 20, 100)
MIN_N = 30
N_BOOT = 2000
RNG = np.random.default_rng(0)


def boot_ci_mean(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(INP)
    m = m[m["aggr"].isin(["buy_aggr", "sell_aggr"])].copy()
    m["U"] = m["agg_party"].astype(str)
    m["side"] = m["agg_role"].astype(str)

    rows: list[dict] = []
    for k in KS:
        col = f"fwd_{k}_sym"
        for (u, side, prod, sprb), g in m.groupby(["U", "side", "product", "spr_b"]):
            x = g[col].dropna().to_numpy(dtype=float)
            n = int(len(x))
            if n < MIN_N or not np.isfinite(x).all():
                continue
            mean = float(x.mean())
            med = float(np.median(x))
            tstat, pval = stats.ttest_1samp(x, 0.0, nan_policy="omit")
            lo, hi = boot_ci_mean(x)
            rows.append(
                {
                    "K": k,
                    "U": u,
                    "aggressor_side": side,
                    "product": prod,
                    "spread_bucket": sprb,
                    "n": n,
                    "mean": mean,
                    "median": med,
                    "tstat_vs0": float(tstat),
                    "pvalue_vs0": float(pval),
                    "frac_pos": float((x > 0).mean()),
                    "boot_mean_ci95_lo": lo,
                    "boot_mean_ci95_hi": hi,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["K", "n"], ascending=[True, False])
    p = OUT / "r4_phase1_aggressor_cell_stats_pooled.csv"
    df.to_csv(p, index=False)
    (OUT / "r4_phase1_supplement_index.json").write_text(
        json.dumps(
            {
                "source_csv": str(INP),
                "n_cells_written": int(len(df)),
                "K_values": list(KS),
                "min_n": MIN_N,
                "n_bootstrap": N_BOOT,
                "output_csv": str(p),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(df)} rows to {p}")


if __name__ == "__main__":
    main()
