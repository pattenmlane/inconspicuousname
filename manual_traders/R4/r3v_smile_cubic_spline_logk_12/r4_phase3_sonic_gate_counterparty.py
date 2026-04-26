#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (5200 & 5300 BBO spread <= 2) × counterparty × spreads.

1) Re-score Phase-1 trade markouts by joint_gate_tight (already in wide CSV).
2) Three-way: (buyer, seller, joint_gate_tight) × symbol for mark_20_same.
3) inclineGod: spread–spread and spread vs extract mid on aligned panel (inner join 5200,5300,EXTRACT per R3 script).
4) Compare Mark01→Mark22 extract mark_20 when gate TRUE vs FALSE.

Outputs in analysis_outputs/:
  r4_phase3_mark20_by_gate_pair_symbol.csv
  r4_phase3_spread_correlations_by_day.csv
  r4_phase3_spread_scatter_data.csv (subsample for plotting)
  r4_phase3_summary.txt

Run: python3 manual_traders/R4/r3v_smile_cubic_spline_logk_12/r4_phase3_sonic_gate_counterparty.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
P1 = OUT
DAYS = (1, 2, 3)
TH = 2
EX = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"


def one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = df[df["product"] == product].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return pd.DataFrame(
        {
            "timestamp": v["timestamp"].values,
            "spread": (ask - bid).astype(float),
            "mid": mid.astype(float),
        }
    )


def aligned_panel(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    a = one_product(df, S5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_product(df, S5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_product(df, EX).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(e, on="timestamp", how="inner")
    m["day"] = day
    return m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(P1 / "r4_trade_markouts_wide.csv")
    wide["joint_gate_tight"] = wide["joint_gate_tight"].astype(bool)

    g3 = (
        wide.groupby(["buyer", "seller", "joint_gate_tight", "symbol"])["mark_20_same"]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
    )
    g3.to_csv(OUT / "r4_phase3_mark20_by_gate_pair_symbol.csv", index=False)

    # Mark 01 -> Mark 22 on extract (often zero rows — basket is on vouchers)
    sub = wide[(wide["symbol"] == EX) & (wide["buyer"] == "Mark 01") & (wide["seller"] == "Mark 22")]
    if len(sub):
        m01 = sub.groupby(["day", "joint_gate_tight"])["mark_20_same"].agg(["count", "mean"]).reset_index()
    else:
        m01 = pd.DataFrame(columns=["day", "joint_gate_tight", "count", "mean"])
    m01.to_csv(OUT / "r4_phase3_m01_m22_extract_mark20_by_gate_day.csv", index=False)

    for vsym in ("VEV_5300", "VEV_5400", "VEV_5500"):
        sv = wide[(wide["symbol"] == vsym) & (wide["buyer"] == "Mark 01") & (wide["seller"] == "Mark 22")]
        if len(sv):
            sv.groupby(["day", "joint_gate_tight"])["mark_20_same"].agg(["count", "mean"]).reset_index().to_csv(
                OUT / f"r4_phase3_m01_m22_{vsym}_mark20_by_gate_day.csv", index=False
            )

    # Mark 67 -> Mark 49 extract
    m67 = wide[
        (wide["symbol"] == EX) & (wide["buyer"] == "Mark 67") & (wide["seller"] == "Mark 49")
    ].groupby(["day", "joint_gate_tight"])["mark_20_same"].agg(["count", "mean"]).reset_index()
    m67.to_csv(OUT / "r4_phase3_m67_m49_extract_mark20_by_gate_day.csv", index=False)

    # Spread–spread correlations (full panel rows, subsample for scatter file)
    corr_rows = []
    scatter_parts = []
    for d in DAYS:
        p = aligned_panel(d)
        for a, b in [("s5200", "s5300"), ("s5200", "s_ext"), ("s5300", "s_ext")]:
            x, y = p[a].to_numpy(), p[b].to_numpy()
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 30:
                continue
            c = float(np.corrcoef(x[m], y[m])[0, 1])
            corr_rows.append({"day": d, "x": a, "y": b, "corr": c, "n": int(m.sum())})
        samp = p.sample(min(5000, len(p)), random_state=d) if len(p) > 0 else p
        samp["day"] = d
        scatter_parts.append(samp[["day", "timestamp", "s5200", "s5300", "s_ext", "m_ext"]])
    pd.DataFrame(corr_rows).to_csv(OUT / "r4_phase3_spread_correlations_by_day.csv", index=False)
    pd.concat(scatter_parts, ignore_index=True).to_csv(OUT / "r4_phase3_spread_scatter_data.csv", index=False)

    # Gate tight fraction on panel
    lines = ["=== Round 4 Phase 3 (Sonic gate × counterparty) ===", ""]
    for d in DAYS:
        p = aligned_panel(d)
        tight = (p["s5200"] <= TH) & (p["s5300"] <= TH)
        lines.append(
            f"day {d}: P(tight)={tight.mean():.4f} n={len(p)} corr(s5200,s5300)={p['s5200'].corr(p['s5300']):.4f}"
        )
    lines.append("")
    lines.append("Mark01→Mark22 on VELVETFRUIT_EXTRACT: n=0 in tape (basket flow is on vouchers).")
    for vsym in ("VEV_5300", "VEV_5400", "VEV_5500"):
        sv = wide[(wide["symbol"] == vsym) & (wide["buyer"] == "Mark 01") & (wide["seller"] == "Mark 22")]
        if len(sv):
            lines.append(f"M01→M22 {vsym} pooled: n={len(sv)} mean_m20={sv['mark_20_same'].mean():.4f} by_gate:")
            for gval, grp in sv.groupby("joint_gate_tight"):
                lines.append(f"    gate={gval} n={len(grp)} mean={grp['mark_20_same'].mean():.4f}")
    lines.append("")
    lines.append("Mark67→Mark49 EXTRACT mark_20 by gate:")
    sub67 = wide[(wide["symbol"] == EX) & (wide["buyer"] == "Mark 67") & (wide["seller"] == "Mark 49")]
    for gval, grp in sub67.groupby("joint_gate_tight"):
        lines.append(f"  gate={gval} n={len(grp)} mean={grp['mark_20_same'].mean():.4f}")
    (OUT / "r4_phase3_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
