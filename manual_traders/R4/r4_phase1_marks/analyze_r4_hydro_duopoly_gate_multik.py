#!/usr/bin/env python3
"""
Round 4 — **Mark 14 → Mark 38** on **HYDROGEL_PACK**: joint tight vs loose mean
**fwd_EXTRACT_K** for K in {5, 20, 100} (Phase 1 bar convention).

Complements ``phase8_*`` (which used K=20 only for the full pair grid).

Run: python3 manual_traders/R4/r4_phase1_marks/analyze_r4_hydro_duopoly_gate_multik.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve()
OUT = HERE.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
KS = (5, 20, 100)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(np.mean(a)), float(np.mean(b)), float(r.statistic), float(r.pvalue)


def main() -> None:
    spec = importlib.util.spec_from_file_location("p3", HERE.parent / "analyze_phase3.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    p1 = mod.load_p1()
    m = mod.merge_trades_with_tight(p1)
    m["tight"] = m["tight"].fillna(False).astype(bool)
    sub = m[
        (m["buyer"] == "Mark 14")
        & (m["seller"] == "Mark 38")
        & (m["symbol"] == "HYDROGEL_PACK")
    ].copy()
    lines = [
        "Mark 14 -> Mark 38, HYDROGEL_PACK: mean fwd_EXTRACT_K | tight vs loose\n",
        f"n total prints with gate flag: {len(sub)}\n\n",
    ]
    rows = []
    for K in KS:
        col = f"fwd_EXTRACT_{K}"
        if col not in sub.columns:
            lines.append(f"  K={K}: column missing\n")
            continue
        xt = sub.loc[sub["tight"], col].astype(float).dropna().values
        xn = sub.loc[~sub["tight"], col].astype(float).dropna().values
        mt, mn, tstat, pval = welch(xt, xn)
        lines.append(
            f"K={K}: n_tight={len(xt)} n_loose={len(xn)} "
            f"mean_tight={mt:.5g} mean_loose={mn:.5g} delta={mt-mn:.5g} "
            f"Welch_t={tstat:.4f} p={pval:.4g}\n"
        )
        rows.append(
            {
                "buyer": "Mark 14",
                "seller": "Mark 38",
                "symbol": "HYDROGEL_PACK",
                "K": K,
                "n_tight": len(xt),
                "n_loose": len(xn),
                "mean_tight": mt,
                "mean_loose": mn,
                "delta_tight_minus_loose": mt - mn,
                "welch_t": tstat,
                "welch_p": pval,
            }
        )
    (OUT / "phase9_hydro_14_38_gate_multik_extract_fwd.txt").write_text("".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "phase9_hydro_14_38_gate_multik_extract_fwd.csv", index=False)
    print("Wrote", OUT / "phase9_hydro_14_38_gate_multik_extract_fwd.txt")


if __name__ == "__main__":
    main()
