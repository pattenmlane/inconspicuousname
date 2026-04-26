#!/usr/bin/env python3
"""
Round 4 — **post-phase** counterparty × **Sonic joint gate** stability.

Uses the same enriched trades + aligned gate panel as ``analyze_phase3.py``
(``merge_trades_with_tight``). For each (buyer, seller, symbol) with enough prints,
compares **mean forward extract mid** (K=20 bars, trade-time convention from
Phase 1) on prints where **joint tight** vs **not tight**, **per day** and pooled.

Outputs (under ``outputs/``):
  - ``phase8_pair_gate_extract_fwd20_by_day.csv`` — per (pair,symbol,day) cells
  - ``phase8_pair_gate_extract_fwd20_pooled.csv`` — pooled over days 1–3
  - ``phase8_top_pairs_gate_summary.txt`` — human-readable top pairs + day stability

Run from repo root:
  python3 manual_traders/R4/r4_phase1_marks/analyze_r4_gate_pair_stability.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
OUT = HERE.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
FWD = "fwd_EXTRACT_20"
MIN_CELL = 8  # min prints in a (day,gate) bucket to report mean
MIN_PAIR_TOTAL = 50  # min pooled prints for (buyer,seller,symbol)
SYMS_FOCUS = [
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
]


def load_phase3():
    spec = importlib.util.spec_from_file_location("p3", HERE.parent / "analyze_phase3.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(r.statistic), float(r.pvalue)


def main() -> None:
    p3 = load_phase3()
    p1 = p3.load_p1()
    m = p3.merge_trades_with_tight(p1)
    m = m[m["symbol"].isin(SYMS_FOCUS)].copy()
    m["tight"] = m["tight"].fillna(False).astype(bool)
    y = m[FWD].astype(float)

    rows_day: list[dict] = []
    for day in DAYS:
        md = m[m["day"] == day]
        for (b, s, sym), g in md.groupby(["buyer", "seller", "symbol"], sort=False):
            gt = g[g["tight"]]
            gn = g[~g["tight"]]
            xt = gt[FWD].astype(float).dropna().values
            xn = gn[FWD].astype(float).dropna().values
            if len(xt) < MIN_CELL or len(xn) < MIN_CELL:
                continue
            mt, mn = float(np.mean(xt)), float(np.mean(xn))
            tstat, pval = welch(xt, xn)
            rows_day.append(
                {
                    "day": day,
                    "buyer": b,
                    "seller": s,
                    "symbol": sym,
                    "n_tight": len(xt),
                    "n_loose": len(xn),
                    "mean_fwd_EXTRACT_20_tight": mt,
                    "mean_fwd_EXTRACT_20_loose": mn,
                    "delta_tight_minus_loose": mt - mn,
                    "welch_t_tight_vs_loose": tstat,
                    "welch_p": pval,
                }
            )

    df_day = pd.DataFrame(rows_day)
    df_day.to_csv(OUT / "phase8_pair_gate_extract_fwd20_by_day.csv", index=False)

    rows_pool: list[dict] = []
    for (b, s, sym), g in m.groupby(["buyer", "seller", "symbol"], sort=False):
        if len(g) < MIN_PAIR_TOTAL:
            continue
        gt = g[g["tight"]]
        gn = g[~g["tight"]]
        xt = gt[FWD].astype(float).dropna().values
        xn = gn[FWD].astype(float).dropna().values
        if len(xt) < MIN_CELL or len(xn) < MIN_CELL:
            continue
        mt, mn = float(np.mean(xt)), float(np.mean(xn))
        tstat, pval = welch(xt, xn)
        days_hit = sorted(g["day"].unique().tolist())
        rows_pool.append(
            {
                "buyer": b,
                "seller": s,
                "symbol": sym,
                "n_tight": len(xt),
                "n_loose": len(xn),
                "mean_fwd_EXTRACT_20_tight": mt,
                "mean_fwd_EXTRACT_20_loose": mn,
                "delta_tight_minus_loose": mt - mn,
                "welch_t_tight_vs_loose": tstat,
                "welch_p": pval,
                "days_present": ",".join(str(int(d)) for d in days_hit),
            }
        )

    df_pool = pd.DataFrame(rows_pool)
    df_pool = df_pool.sort_values("welch_p", na_position="last")
    df_pool.to_csv(OUT / "phase8_pair_gate_extract_fwd20_pooled.csv", index=False)

    # Day-stability: pairs that appear in df_day for all 3 days with same sign of delta
    lines: list[str] = []
    lines.append(
        "Round 4 — joint gate × (buyer,seller,symbol): fwd_EXTRACT_20 tight vs loose\n"
        f"MIN_CELL={MIN_CELL} MIN_PAIR_TOTAL={MIN_PAIR_TOTAL}\n\n"
    )
    top = df_pool.head(25)
    lines.append("Top 25 by pooled Welch p (smallest p first):\n")
    for _, r in top.iterrows():
        lines.append(
            f"  {r['buyer']}->{r['seller']} {r['symbol']}: "
            f"nT={int(r['n_tight'])} nL={int(r['n_loose'])} "
            f"meanT={r['mean_fwd_EXTRACT_20_tight']:.4f} meanL={r['mean_fwd_EXTRACT_20_loose']:.4f} "
            f"delta={r['delta_tight_minus_loose']:.4f} t={r['welch_t_tight_vs_loose']:.3f} p={r['welch_p']:.3g} "
            f"days={r['days_present']}\n"
        )

    if len(df_day):
        sig = df_day[df_day["welch_p"] < 0.05].copy()
        lines.append(f"\nPer-day cells with Welch p<0.05: {len(sig)} (multiple testing: descriptive only)\n")
        vc = sig["symbol"].value_counts()
        lines.append("Breakdown by symbol (count of significant day-cells):\n")
        for sym, c in vc.items():
            lines.append(f"  {sym}: {int(c)}\n")

        # Same (b,s,sym) across days: sign of delta
        lines.append("\nSign consistency across days (only keys with 3 day-rows in csv):\n")
        for key, g2 in df_day.groupby(["buyer", "seller", "symbol"]):
            if len(g2) != 3:
                continue
            deltas = np.sign(g2["delta_tight_minus_loose"].values)
            if np.all(deltas > 0) or np.all(deltas < 0):
                lines.append(
                    f"  SAME_SIGN_3D {key[0]}->{key[1]} {key[2]}: "
                    f"deltas {g2['delta_tight_minus_loose'].round(4).tolist()} "
                    f"p {[float(x) for x in g2['welch_p']]}\n"
                )

    lines.append(
        "\nInterpretation: large |delta| with small p on **one** day and mixed signs "
        "across days is weak evidence of a tradeable Mark×gate rule; three-day tape "
        "is too short for strong LODO. Compare to Phase 3 panel Welch (price bars, "
        "not trade-conditioned).\n"
    )
    (OUT / "phase8_top_pairs_gate_summary.txt").write_text("".join(lines), encoding="utf-8")
    print("Wrote", OUT / "phase8_pair_gate_extract_fwd20_by_day.csv")
    print("Wrote", OUT / "phase8_pair_gate_extract_fwd20_pooled.csv")
    print("Wrote", OUT / "phase8_top_pairs_gate_summary.txt")


if __name__ == "__main__":
    main()
