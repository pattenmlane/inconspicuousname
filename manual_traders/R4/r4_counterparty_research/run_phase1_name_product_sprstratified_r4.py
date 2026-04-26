#!/usr/bin/env python3
"""
Phase 1 bullet 1 supplement: participant x product x spr_regime cell stats for K=20 markouts.
Roles: aggr_buy (group buyer), aggr_sell (group seller). Minimum n per cell = 20 for CSV export.
Columns: t-stat one-sample test mean(mark_20_u) != 0, same for mark_20_sym ( scipy ttest_1samp ).

Input: r4_trades_enriched_markouts.csv
Output: r4_phase1_name_product_sprreg_mark20.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ENR = Path(__file__).resolve().parent / "outputs" / "r4_trades_enriched_markouts.csv"
OUT = Path(__file__).resolve().parent / "outputs"
MIN_N = 20


def tstat1(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    return float(stats.ttest_1samp(x, 0, nan_policy="omit").statistic)


def pval1(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    return float(stats.ttest_1samp(x, 0, nan_policy="omit").pvalue)


def main() -> None:
    df = pd.read_csv(ENR)
    for c in "mark_20_sym", "mark_20_u", "spr_regime", "symbol", "aggressor", "buyer", "seller":
        if c in ("spr_regime", "symbol", "aggressor", "buyer", "seller"):
            df[c] = df[c].astype(str)
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows: list[dict] = []
    for role, gcol, mask in [
        ("aggr_buy", "buyer", df["aggressor"] == "buy"),
        ("aggr_sell", "seller", df["aggressor"] == "sell"),
    ]:
        sub = df[mask]
        for (name, sym, reg), g in sub.groupby([gcol, "symbol", "spr_regime"]):
            if str(name) in ("nan", "None", ""):
                continue
            symv = g["mark_20_sym"].dropna()
            uu = g["mark_20_u"].dropna()
            if len(uu) < MIN_N:
                continue
            rows.append(
                {
                    "role": role,
                    "name": str(name),
                    "symbol": str(sym),
                    "spr_regime": str(reg),
                    "n": int(len(g)),
                    "n_u20": int(len(uu)),
                    "mean_u20": float(uu.mean()) if len(uu) else float("nan"),
                    "p_u20_vs0": pval1(uu.to_numpy()),
                    "t_u20": tstat1(uu.to_numpy()),
                    "mean_sym20": float(symv.mean()) if len(symv) else float("nan"),
                    "p_sym20_vs0": pval1(symv.to_numpy()),
                }
            )
    t = pd.DataFrame(rows)
    t = t.sort_values(["n_u20", "p_u20_vs0"], ascending=[False, True])
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "r4_phase1_name_product_sprreg_mark20.csv", index=False)
    t.head(50).to_csv(OUT / "r4_phase1_name_product_sprreg_mark20_top50.csv", index=False)
    print("wrote", OUT / "r4_phase1_name_product_sprreg_mark20.csv", "rows", len(t))


if __name__ == "__main__":
    main()
