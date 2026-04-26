#!/usr/bin/env python3
"""
Round 4 — **HYDROGEL_PACK** duopoly × joint gate: mean **fwd_EXTRACT_K** tight vs loose.

- Directions: **Mark 14 → Mark 38** and **Mark 38 → Mark 14**
- Horizons K in {5, 20, 100} (Phase 1 price-bar steps)
- **Pooled** and **per-day** Welch (skip day if n<2 in either bucket)

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
DAYS = (1, 2, 3)
PAIRS = (
    ("Mark 14", "Mark 38"),
    ("Mark 38", "Mark 14"),
)


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(np.mean(a)), float(np.mean(b)), float(r.statistic), float(r.pvalue)


def block_pair_day(
    sub: pd.DataFrame, buyer: str, seller: str, day: int | None, K: int
) -> str:
    col = f"fwd_EXTRACT_{K}"
    g = sub if day is None else sub[sub["day"] == day]
    xt = g.loc[g["tight"], col].astype(float).dropna().values
    xn = g.loc[~g["tight"], col].astype(float).dropna().values
    mt, mn, tstat, pval = welch(xt, xn)
    d = mt - mn if np.isfinite(mt) and np.isfinite(mn) else float("nan")
    day_s = "all" if day is None else str(day)
    return (
        f"{buyer}->{seller} day={day_s} K={K}: nT={len(xt)} nL={len(xn)} "
        f"meanT={mt:.5g} meanL={mn:.5g} delta={d:.5g} Welch_t={tstat:.4f} p={pval:.4g}\n"
    )


def main() -> None:
    spec = importlib.util.spec_from_file_location("p3", HERE.parent / "analyze_phase3.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    p1 = mod.load_p1()
    m = mod.merge_trades_with_tight(p1)
    m["tight"] = m["tight"].fillna(False).astype(bool)

    lines: list[str] = []
    rows_pool: list[dict] = []
    rows_day: list[dict] = []

    for buyer, seller in PAIRS:
        sub = m[
            (m["buyer"] == buyer)
            & (m["seller"] == seller)
            & (m["symbol"] == "HYDROGEL_PACK")
        ].copy()
        lines.append(f"\n=== {buyer} -> {seller}, HYDROGEL_PACK (n={len(sub)}) ===\n")
        for K in KS:
            col = f"fwd_EXTRACT_{K}"
            if col not in sub.columns:
                lines.append(f"  K={K}: missing column\n")
                continue
            lines.append(block_pair_day(sub, buyer, seller, None, K))
            mt, mn, tstat, pval = welch(
                sub.loc[sub["tight"], col].astype(float).dropna().values,
                sub.loc[~sub["tight"], col].astype(float).dropna().values,
            )
            rows_pool.append(
                {
                    "buyer": buyer,
                    "seller": seller,
                    "day": "all",
                    "K": K,
                    "n_tight": int(sub.loc[sub["tight"], col].notna().sum()),
                    "n_loose": int(sub.loc[~sub["tight"], col].notna().sum()),
                    "mean_tight": mt,
                    "mean_loose": mn,
                    "delta": mt - mn,
                    "welch_t": tstat,
                    "welch_p": pval,
                }
            )
            for day in DAYS:
                line = block_pair_day(sub, buyer, seller, day, K)
                lines.append("  " + line)
                gd = sub[sub["day"] == day]
                xt = gd.loc[gd["tight"], col].astype(float).dropna().values
                xn = gd.loc[~gd["tight"], col].astype(float).dropna().values
                mt2, mn2, ts2, pv2 = welch(xt, xn)
                rows_day.append(
                    {
                        "buyer": buyer,
                        "seller": seller,
                        "day": day,
                        "K": K,
                        "n_tight": len(xt),
                        "n_loose": len(xn),
                        "mean_tight": mt2,
                        "mean_loose": mn2,
                        "delta": mt2 - mn2,
                        "welch_t": ts2,
                        "welch_p": pv2,
                    }
                )

    txt_path = OUT / "phase9_hydro_duopoly_gate_multik_extract_fwd.txt"
    txt_path.write_text("HYDRO duopoly × joint gate vs fwd_EXTRACT_K\n" + "".join(lines), encoding="utf-8")
    pd.DataFrame(rows_pool).to_csv(
        OUT / "phase9_hydro_duopoly_gate_multik_extract_fwd_pooled.csv", index=False
    )
    pd.DataFrame(rows_day).to_csv(
        OUT / "phase9_hydro_duopoly_gate_multik_extract_fwd_by_day.csv", index=False
    )
    print("Wrote", txt_path)


if __name__ == "__main__":
    main()
