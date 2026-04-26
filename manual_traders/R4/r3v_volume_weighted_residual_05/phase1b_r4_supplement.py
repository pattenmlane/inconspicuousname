#!/usr/bin/env python3
"""
Round 4 Phase 1 **supplement** — expanded stratification + bootstrap CIs + coverage.

Reads `analysis_outputs/phase1/r4_trades_enriched.csv` (from phase1_r4_counterparty_analysis.py).

Adds:
- `hour_quarter`: within each `day`, quartile bin of `timestamp` (0–3) = coarse session.
- `burst`: 1 if same (day,timestamp) has >=3 trade rows in enriched file.
- `spr_u_bin`: `u1`, `u2_6`, `wide_u` from `spread_u` at print.

Outputs:
- r4_phase1b_hour_quarter_mark67_u.csv — Mark 67 U buy_agg, fwd_u K∈{5,20}, by day × hour_quarter
- r4_phase1b_bootstrap_top_cells.csv — pooled mean + bootstrap 95% CI for key (Mark, role, symbol, K) cells
- r4_phase1b_participant_coverage.txt — distinct names, row counts
- r4_phase1b_cross_asset_mark67_u.csv — Mark 67 U buy_agg: mean fwd_mid vs fwd_u vs fwd_h at K=5,20
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
INP = Path(__file__).resolve().parent / "analysis_outputs" / "phase1" / "r4_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
BOOT = 4000


def u_spread_bin(s: float) -> str:
    if not np.isfinite(s):
        return "na"
    if s <= 1:
        return "u1"
    if s <= 6:
        return "u2_6"
    return "wide_u"


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = BOOT) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    if len(x) == 1:
        return m, m, m
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return m, lo, hi


def main() -> None:
    if not INP.is_file():
        raise SystemExit(f"missing {INP}; run phase1_r4_counterparty_analysis.py first")

    ev = pd.read_csv(INP)
    ev["timestamp"] = ev["timestamp"].astype(int)
    ev["day"] = ev["day"].astype(int)

    # burst: count rows per (day, ts)
    cnt = ev.groupby(["day", "timestamp"]).size().rename("burst_n")
    ev = ev.merge(cnt, on=["day", "timestamp"], how="left")
    ev["burst"] = (ev["burst_n"] >= 3).astype(int)

    # hour quarter within day
    ev["hour_quarter"] = ev.groupby("day")["timestamp"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop").astype(float)
    )
    ev["spr_u_bin"] = ev["spread_u"].apply(u_spread_bin)

    # --- Mark 67 U buy_agg stratification ---
    m67 = ev[
        (ev["symbol"] == "VELVETFRUIT_EXTRACT")
        & (ev["buyer"] == "Mark 67")
        & (ev["aggressor"] == "buy_agg")
    ].copy()
    rows_hq = []
    for (d, hq), g in m67.groupby(["day", "hour_quarter"]):
        for K in (5, 20):
            c = f"fwd_u_{K}"
            x = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy()
            if len(x) < 3:
                continue
            m, lo, hi = bootstrap_mean_ci(x)
            rows_hq.append(
                {
                    "slice": "Mark67|U|buy_agg|fwd_extract",
                    "day": int(d),
                    "hour_quarter": float(hq),
                    "K": K,
                    "n": len(x),
                    "mean": m,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "frac_pos": float((x > 0).mean()),
                }
            )
    pd.DataFrame(rows_hq).to_csv(OUT / "r4_phase1b_hour_quarter_mark67_u.csv", index=False)

    # burst vs not for Mark67 U buy_agg K=20
    m67["fu20"] = pd.to_numeric(m67["fwd_u_20"], errors="coerce")
    br = m67[m67["burst"] == 1]["fu20"].dropna()
    nb = m67[m67["burst"] == 0]["fu20"].dropna()
    burst_cmp = pd.DataFrame(
        [
            {
                "slice": "Mark67|U|buy_agg",
                "K": 20,
                "burst": 1,
                "n": len(br),
                "mean": float(br.mean()) if len(br) else float("nan"),
            },
            {
                "slice": "Mark67|U|buy_agg",
                "K": 20,
                "burst": 0,
                "n": len(nb),
                "mean": float(nb.mean()) if len(nb) else float("nan"),
            },
        ]
    )
    burst_cmp.to_csv(OUT / "r4_phase1b_mark67_u_burst_vs_not_k20.csv", index=False)

    # spr_u_bin for Mark67 U buy_agg K=20
    spr_rows = []
    for b, g in m67.groupby("spr_u_bin"):
        x = pd.to_numeric(g["fwd_u_20"], errors="coerce").dropna().to_numpy()
        if len(x) < 5:
            continue
        m, lo, hi = bootstrap_mean_ci(x)
        spr_rows.append(
            {
                "slice": "Mark67|U|buy_agg|fwd_u",
                "spr_u_bin": b,
                "K": 20,
                "n": len(x),
                "mean": m,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        )
    pd.DataFrame(spr_rows).to_csv(OUT / "r4_phase1b_mark67_u_spreadu_bin_k20.csv", index=False)

    # --- Bootstrap top Phase-1 style cells ---
    cells = []
    specs = [
        ("Mark 67", "VELVETFRUIT_EXTRACT", "buy_agg", "buyer"),
        ("Mark 22", "VELVETFRUIT_EXTRACT", "sell_agg", "seller"),
        ("Mark 49", "VELVETFRUIT_EXTRACT", "sell_agg", "seller"),
        ("Mark 14", "VELVETFRUIT_EXTRACT", "sell_agg", "seller"),
    ]
    for mark, sym, ag, col in specs:
        g = ev[(ev["symbol"] == sym) & (ev["aggressor"] == ag) & (ev[col] == mark)]
        for K in (5, 20, 100):
            c = f"fwd_mid_{K}"
            x = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy()
            if len(x) < 30:
                continue
            m, lo, hi = bootstrap_mean_ci(x)
            cells.append(
                {
                    "Mark": mark,
                    "symbol": sym,
                    "aggressor": ag,
                    "K": K,
                    "n": len(x),
                    "mean_fwd_same_sym": m,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "frac_pos": float((x > 0).mean()),
                }
            )
    pd.DataFrame(cells).to_csv(OUT / "r4_phase1b_bootstrap_top_cells.csv", index=False)

    # Per-day mean for Mark67 U buy_agg fwd_u_20
    daystab = []
    for d in sorted(ev["day"].unique()):
        g = m67[m67["day"] == d]
        x = pd.to_numeric(g["fwd_u_20"], errors="coerce").dropna().to_numpy()
        if len(x) == 0:
            continue
        m, lo, hi = bootstrap_mean_ci(x, n_boot=2000)
        daystab.append({"day": int(d), "n": len(x), "mean_fwd_u_20": m, "ci95_lo": lo, "ci95_hi": hi})
    pd.DataFrame(daystab).to_csv(OUT / "r4_phase1b_mark67_u_day_bootstrap_k20.csv", index=False)

    # Cross-asset Mark67 U buy_agg
    g = m67
    xr = []
    for K in (5, 20):
        for col, label in [
            (f"fwd_mid_{K}", "same_sym_mid"),
            (f"fwd_u_{K}", "extract_mid"),
            (f"fwd_h_{K}", "hydro_mid"),
        ]:
            x = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
            if len(x) < 30:
                continue
            m, lo, hi = bootstrap_mean_ci(x)
            xr.append({"K": K, "fwd_target": label, "n": len(x), "mean": m, "ci95_lo": lo, "ci95_hi": hi})
    pd.DataFrame(xr).to_csv(OUT / "r4_phase1b_cross_asset_mark67_u.csv", index=False)

    # Coverage
    names = sorted(set(ev["buyer"].astype(str)) | set(ev["seller"].astype(str)))
    lines = [
        f"Distinct counterparty strings (buyer ∪ seller): {len(names)}",
        f"Total enriched trade rows: {len(ev)}",
        f"Symbols: {ev['symbol'].nunique()}",
    ]
    (OUT / "r4_phase1b_participant_coverage.txt").write_text("\n".join(lines) + "\n")

    print("Wrote Phase 1b supplement to", OUT)


if __name__ == "__main__":
    main()
