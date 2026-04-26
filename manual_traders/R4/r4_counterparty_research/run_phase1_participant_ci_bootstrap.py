#!/usr/bin/env python3
"""
Phase 1 supplement: bootstrap CI for mean markouts (K=20), per-day means, burst Welch t-test,
Mark 22 VEV aggressive sells by session.

Input: outputs/r4_trades_enriched_markouts.csv (from run_phase1_counterparty_analysis.py).
Horizon: mark_20_* = forward mid change over K=20 price ticks (timestamp + 20*100), Phase 1 convention.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
ENR = Path(__file__).resolve().parent / "outputs" / "r4_trades_enriched_markouts.csv"
OUT = Path(__file__).resolve().parent / "outputs"
RNG = np.random.default_rng(42)
N_BOOT = 4000
MIN_N_BOOT = 100


def boot_ci_mean(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [float(RNG.choice(x, size=len(x), replace=True).mean()) for _ in range(N_BOOT)]
    arr = np.array(means)
    return float(np.mean(x)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    if not ENR.is_file():
        raise SystemExit(f"missing {ENR}")
    df = pd.read_csv(ENR)
    for c in ("mark_20_sym", "mark_20_u", "day"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    boot_rows = []
    for side, grp_col, label in [
        ("buy", "buyer", "aggressive_buy"),
        ("sell", "seller", "aggressive_sell"),
    ]:
        sub = df[df["aggressor"] == side]
        for name, g in sub.groupby(grp_col):
            nm = str(name)
            if nm in ("nan", "None"):
                continue
            ms = g["mark_20_sym"].dropna().to_numpy()
            mu = g["mark_20_u"].dropna().to_numpy()
            if len(mu) < MIN_N_BOOT:
                continue
            m_sym, lo_s, hi_s = boot_ci_mean(ms) if len(ms) >= MIN_N_BOOT else (float("nan"),) * 3
            m_u, lo_u, hi_u = boot_ci_mean(mu)
            boot_rows.append(
                {
                    "side": label,
                    "name": nm,
                    "n_sym": int(len(ms)),
                    "mean_mark20_sym": m_sym,
                    "ci95_lo_sym": lo_s,
                    "ci95_hi_sym": hi_s,
                    "n_u": int(len(mu)),
                    "mean_mark20_u": m_u,
                    "ci95_lo_u": lo_u,
                    "ci95_hi_u": hi_u,
                }
            )

    pd.DataFrame(boot_rows).sort_values(["side", "n_u"], ascending=[True, False]).to_csv(
        OUT / "r4_phase1_participant_u20_bootstrap.csv", index=False
    )

    # per-day means for names with pooled n>=200 on mark_20_u
    names_big = {r["name"] for r in boot_rows if r["n_u"] >= 200}
    by_day_rows = []
    for side, grp_col, label in [("buy", "buyer", "aggressive_buy"), ("sell", "seller", "aggressive_sell")]:
        sub = df[df["aggressor"] == side]
        for nm in names_big:
            g = sub[sub[grp_col].astype(str) == nm]
            for d, gd in g.groupby("day"):
                mu = gd["mark_20_u"].dropna()
                if len(mu) >= 20:
                    by_day_rows.append(
                        {
                            "side": label,
                            "name": nm,
                            "day": int(d),
                            "n": int(len(mu)),
                            "mean_mark20_u": float(mu.mean()),
                        }
                    )
    pd.DataFrame(by_day_rows).to_csv(OUT / "r4_phase1_participant_u20_by_day.csv", index=False)

    # burst vs non Welch on mark_20_u (reuse logic: merge burst flag from trades)
    DATA = REPO / "Prosperity4Data" / "ROUND_4"
    tr_parts = []
    for day in (1, 2, 3):
        t = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        t["day"] = day
        tr_parts.append(t)
    tr = pd.concat(tr_parts, ignore_index=True)
    burst_n = tr.groupby(["day", "timestamp"]).size().rename("n_prints").reset_index()
    burst_n["burst"] = burst_n["n_prints"] >= 4
    enb = df.merge(burst_n[["day", "timestamp", "burst"]], on=["day", "timestamp"], how="left")
    enb["burst"] = enb["burst"].fillna(False)
    a = enb[enb["burst"]]["mark_20_u"].dropna().to_numpy()
    b = enb[~enb["burst"]]["mark_20_u"].dropna().to_numpy()
    tt = stats.ttest_ind(a, b, equal_var=False)
    welch = {
        "n_burst": int(len(a)),
        "n_non": int(len(b)),
        "mean_burst": float(np.mean(a)),
        "mean_non": float(np.mean(b)),
        "welch_statistic": float(tt.statistic),
        "welch_pvalue_two_sided": float(tt.pvalue),
    }
    (OUT / "r4_phase1_burst_vs_non_welch.json").write_text(json.dumps(welch, indent=2), encoding="utf-8")

    # Mark 22 aggressive sells on VEV only, by session
    vev = df["symbol"].astype(str).str.startswith("VEV_")
    m22s = (
        df[vev & (df["aggressor"] == "sell") & (df["seller"].astype(str) == "Mark 22")]
        .groupby("session")["mark_20_u"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    m22s.to_csv(OUT / "r4_phase1_mark22_vev_sell_by_session.csv", index=False)

    summary = {
        "bootstrap_replicates": N_BOOT,
        "min_n_for_bootstrap_table": MIN_N_BOOT,
        "participant_bootstrap_csv": str(OUT / "r4_phase1_participant_u20_bootstrap.csv"),
        "by_day_csv": str(OUT / "r4_phase1_participant_u20_by_day.csv"),
        "welch_burst": welch,
        "mark22_vev_sell_session_csv": str(OUT / "r4_phase1_mark22_vev_sell_by_session.csv"),
        "notes": [
            "Mark 67 aggressive buy: check if CI for mark_20_sym excludes 0.",
            "Mark 22 aggressive sell: CI for mark_20_sym short-horizon option markout.",
        ],
    }
    (OUT / "r4_phase1_participant_u20_bootstrap.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT / "r4_phase1_participant_u20_bootstrap.json")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
