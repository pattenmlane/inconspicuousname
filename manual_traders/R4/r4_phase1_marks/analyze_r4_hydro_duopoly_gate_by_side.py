#!/usr/bin/env python3
"""
Round 4 — **HYDROGEL_PACK** Mark14↔Mark38: joint **tight vs loose** on **fwd_EXTRACT_20**,
stratified by **aggressor side** (Phase 1: ``side`` = aggr_buy / aggr_sell / at_mid).

Adverse-selection style read: if only **aggr_buy** (Mark 14 lifting offer) shows the
tight>loose lift, passive **aggr_sell** prints may be a different story.

Outputs:
  - ``outputs/phase11_hydro_duopoly_gate_by_side_pooled.csv``
  - ``outputs/phase11_hydro_duopoly_gate_by_side_by_day.csv``
  - ``outputs/phase11_hydro_duopoly_gate_by_side_summary.txt``

Run: python3 manual_traders/R4/r4_phase1_marks/analyze_r4_hydro_duopoly_gate_by_side.py
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
FWD = "fwd_EXTRACT_20"
DAYS = (1, 2, 3)
PAIRS = (("Mark 14", "Mark 38"), ("Mark 38", "Mark 14"))


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"),) * 4
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(np.mean(a)), float(np.mean(b)), float(r.statistic), float(r.pvalue)


def one_cell(xt: np.ndarray, xn: np.ndarray) -> dict:
    mt, mn, tstat, pval = welch(xt, xn)
    return {
        "n_tight": len(xt),
        "n_loose": len(xn),
        "mean_tight": mt,
        "mean_loose": mn,
        "delta": mt - mn if np.isfinite(mt) and np.isfinite(mn) else float("nan"),
        "welch_t": tstat,
        "welch_p": pval,
    }


def main() -> None:
    spec = importlib.util.spec_from_file_location("p3", HERE.parent / "analyze_phase3.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    p1 = mod.load_p1()
    m = mod.merge_trades_with_tight(p1)
    m["tight"] = m["tight"].fillna(False).astype(bool)
    m = m[m["symbol"] == "HYDROGEL_PACK"].copy()

    rows_pool: list[dict] = []
    rows_day: list[dict] = []
    lines: list[str] = []

    for buyer, seller in PAIRS:
        sub = m[(m["buyer"] == buyer) & (m["seller"] == seller)].copy()
        lines.append(f"\n=== {buyer} -> {seller} (n={len(sub)}) ===\n")
        for side in sorted(sub["side"].dropna().unique()):
            s2 = sub[sub["side"] == side]
            if len(s2) < 10:
                lines.append(f"  side={side}: skip (n={len(s2)})\n")
                continue
            xt = s2.loc[s2["tight"], FWD].astype(float).dropna().values
            xn = s2.loc[~s2["tight"], FWD].astype(float).dropna().values
            c = one_cell(xt, xn)
            lines.append(
                f"  pooled side={side}: nT={c['n_tight']} nL={c['n_loose']} "
                f"delta={c['delta']:.4g} t={c['welch_t']:.3f} p={c['welch_p']:.3g}\n"
            )
            row = {
                "buyer": buyer,
                "seller": seller,
                "side": side,
                "day": "all",
                **c,
            }
            rows_pool.append(row)
            for day in DAYS:
                d = s2[s2["day"] == day]
                xt2 = d.loc[d["tight"], FWD].astype(float).dropna().values
                xn2 = d.loc[~d["tight"], FWD].astype(float).dropna().values
                c2 = one_cell(xt2, xn2)
                rows_day.append(
                    {
                        "buyer": buyer,
                        "seller": seller,
                        "side": side,
                        "day": day,
                        **c2,
                    }
                )
                lines.append(
                    f"    day {day} side={side}: nT={c2['n_tight']} nL={c2['n_loose']} "
                    f"delta={c2['delta']:.4g} p={c2['welch_p']:.3g}\n"
                )

    txt = OUT / "phase11_hydro_duopoly_gate_by_side_summary.txt"
    txt.write_text(
        "HYDRO duopoly × joint gate × aggressor side (fwd_EXTRACT_20)\n" + "".join(lines),
        encoding="utf-8",
    )
    pd.DataFrame(rows_pool).to_csv(OUT / "phase11_hydro_duopoly_gate_by_side_pooled.csv", index=False)
    pd.DataFrame(rows_day).to_csv(OUT / "phase11_hydro_duopoly_gate_by_side_by_day.csv", index=False)
    print("Wrote", txt)


if __name__ == "__main__":
    main()
