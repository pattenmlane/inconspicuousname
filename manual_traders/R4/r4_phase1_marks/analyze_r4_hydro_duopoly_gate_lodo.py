#!/usr/bin/env python3
"""
Round 4 — **LODO** (leave-one-day-out) for **HYDROGEL_PACK** Mark14↔Mark38 × joint gate.

Holds out day **D** and recomputes **tight vs loose** mean **fwd_EXTRACT_20** (Welch)
on the **other two** days only. Also writes **full 3-day** pooled (same pairs as
phase8/11).

Outputs:
  - ``outputs/phase12_hydro_duopoly_gate_lodo_pooled3d.csv``
  - ``outputs/phase12_hydro_duopoly_gate_lodo.csv`` (2-day trains)
  - ``outputs/phase12_hydro_duopoly_gate_lodo.txt``

Run: python3 manual_traders/R4/r4_phase1_marks/analyze_r4_hydro_duopoly_gate_lodo.py
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
DAYS = [1, 2, 3]
FWD = "fwd_EXTRACT_20"
PAIRS = (("Mark 14", "Mark 38"), ("Mark 38", "Mark 14"))


def welch(xt: np.ndarray, xn: np.ndarray) -> tuple[float, float, float, float, int, int]:
    a = np.asarray(xt, dtype=float)
    b = np.asarray(xn, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    nta, nlo = len(a), len(b)
    if nta < 2 or nlo < 2:
        return (float("nan"),) * 4 + (nta, nlo)
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(np.mean(a)), float(np.mean(b)), float(r.statistic), float(r.pvalue), nta, nlo


def sub_pair(m: pd.DataFrame, buyer: str, seller: str) -> pd.DataFrame:
    return m[(m["buyer"] == buyer) & (m["seller"] == seller) & (m["symbol"] == "HYDROGEL_PACK")].copy()


def main() -> None:
    spec = importlib.util.spec_from_file_location("p3", HERE.parent / "analyze_phase3.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    p1 = mod.load_p1()
    m = mod.merge_trades_with_tight(p1)
    m["tight"] = m["tight"].fillna(False).astype(bool)

    rows_pool = []
    for buyer, seller in PAIRS:
        g = sub_pair(m, buyer, seller)
        xt = g.loc[g["tight"], FWD].astype(float).dropna().values
        xn = g.loc[~g["tight"], FWD].astype(float).dropna().values
        mt, mn, tstat, pval, nta, nlo = welch(xt, xn)
        rows_pool.append(
            {
                "buyer": buyer,
                "seller": seller,
                "holdout": "none (pooled 3d)",
                "train_days": "1,2,3",
                "n_tight": nta,
                "n_loose": nlo,
                "mean_tight": mt,
                "mean_loose": mn,
                "delta_tight_minus_loose": (mt - mn) if np.isfinite(mt) and np.isfinite(mn) else float("nan"),
                "welch_t": tstat,
                "welch_p": pval,
            }
        )
    full_df = pd.DataFrame(rows_pool)
    full_df.to_csv(OUT / "phase12_hydro_duopoly_gate_lodo_pooled3d.csv", index=False)

    rows_lodo = []
    for buyer, seller in PAIRS:
        g = sub_pair(m, buyer, seller)
        for d_hold in DAYS:
            tr = g[g["day"] != d_hold]
            tr_t = tr[tr["tight"]]
            tr_f = tr[~tr["tight"]]
            xt = tr_t[FWD].astype(float).dropna().values
            xn = tr_f[FWD].astype(float).dropna().values
            mt, mn, tstat, pval, nta, nlo = welch(xt, xn)
            tdays = [str(x) for x in DAYS if x != d_hold]
            rows_lodo.append(
                {
                    "buyer": buyer,
                    "seller": seller,
                    "holdout_day": d_hold,
                    "train_days": ",".join(tdays),
                    "n_tight": nta,
                    "n_loose": nlo,
                    "mean_tight": mt,
                    "mean_loose": mn,
                    "delta_tight_minus_loose": (mt - mn) if np.isfinite(mt) and np.isfinite(mn) else float("nan"),
                    "welch_t": tstat,
                    "welch_p": pval,
                }
            )
    lodo_df = pd.DataFrame(rows_lodo)
    lodo_df.to_csv(OUT / "phase12_hydro_duopoly_gate_lodo.csv", index=False)

    txt = OUT / "phase12_hydro_duopoly_gate_lodo.txt"
    txt.write_text(
        "LODO: 2-day train Welch (fwd_EXTRACT_20 tight vs loose), hydro duopoly only\n\n"
        "POOLED 3d:\n"
        + full_df.to_string(index=False)
        + "\n\n2-DAY TRAINS:\n"
        + lodo_df.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    print("Wrote", txt)


if __name__ == "__main__":
    main()
