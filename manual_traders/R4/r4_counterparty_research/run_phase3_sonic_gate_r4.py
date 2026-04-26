#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 spread <= 2) on ROUND_4 tape,
inclineGod-style spread–spread / spread–mid correlations, and counterparty x gate interactions.

Convention: same as round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
inner-join timestamps where VEV_5200, VEV_5300, and VELVETFRUIT_EXTRACT rows exist; spread = ask1-bid1.
Forward extract move K=20 uses shift(-20) on the aligned panel (20 forward *rows*, not wall-clock).

Inputs: Prosperity4Data/ROUND_4/prices_round_4_day_{1,2,3}.csv
        manual_traders/R4/r4_counterparty_research/outputs/r4_trades_enriched_markouts.csv
Outputs: manual_traders/R4/r4_counterparty_research/outputs/r4_phase3_*
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
ENR = OUT / "r4_trades_enriched_markouts.csv"
TH = 2
K = 20
DAYS = (1, 2, 3)


def one_spread(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates("timestamp", keep="first")
        .sort_values("timestamp")
    )
    b = pd.to_numeric(v["bid_price_1"], errors="coerce")
    a = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    day_col = v["day"] if "day" in v.columns else int(df["day"].iloc[0])
    days = v["day"].astype(int) if "day" in v.columns else int(day_col)
    return pd.DataFrame(
        {
            "day": days,
            "timestamp": v["timestamp"].astype(int),
            "spread": (a - b).astype(float),
            "mid": mid.astype(float),
        }
    )


def aligned_day(df: pd.DataFrame) -> pd.DataFrame:
    a = one_spread(df, "VEV_5200").rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_spread(df, "VEV_5300").rename(columns={"spread": "s5300", "mid": "mid5300"})
    u = one_spread(df, "VELVETFRUIT_EXTRACT").rename(columns={"spread": "s_u", "mid": "m_u"})
    h = one_spread(df, "HYDROGEL_PACK").rename(columns={"spread": "s_h", "mid": "m_h"})
    m = a.merge(b, on=["day", "timestamp"], how="inner").merge(
        u[["day", "timestamp", "s_u", "m_u"]], on=["day", "timestamp"], how="inner"
    )
    m = m.merge(h[["day", "timestamp", "s_h", "m_h"]], on=["day", "timestamp"], how="inner")
    m = m.sort_values(["day", "timestamp"]).reset_index(drop=True)
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    for kk in (5, 20, 100):
        m[f"m_u_f_{kk}"] = m.groupby("day")["m_u"].shift(-kk)
        m[f"fwd_u_{kk}"] = m[f"m_u_f_{kk}"] - m["m_u"]
    # backward-compatible name used below
    m["fwd_u_k"] = m["fwd_u_20"]
    return m


def main() -> None:
    parts = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        parts.append(aligned_day(df))
    panel = pd.concat(parts, ignore_index=True)

    # --- inclineGod: spread–spread / spread vs mid (extract) ---
    cor_rows = []
    for d, g in panel.groupby("day"):
        cor = {
            "day": int(d),
            "corr_s5200_s5300": float(g["s5200"].corr(g["s5300"])),
            "corr_s5200_s_u": float(g["s5200"].corr(g["s_u"])),
            "corr_s5300_s_u": float(g["s5300"].corr(g["s_u"])),
            "corr_s5200_m_u": float(g["s5200"].corr(g["m_u"])),
            "corr_s5300_m_u": float(g["s5300"].corr(g["m_u"])),
            "corr_s5200_s_h": float(g["s5200"].corr(g["s_h"])),
            "corr_s5300_s_h": float(g["s5300"].corr(g["s_h"])),
            "corr_s_u_s_h": float(g["s_u"].corr(g["s_h"])),
            "n": len(g),
        }
        cor_rows.append(cor)
    pd.DataFrame(cor_rows).to_csv(OUT / "r4_phase3_spread_correlations_by_day.csv", index=False)

    valid = panel["fwd_u_20"].notna()
    pv = panel.loc[valid]
    tmask = pv["tight"]
    by_k = {}
    for kk in (5, 20, 100):
        col = f"fwd_u_{kk}"
        vv = panel.loc[panel[col].notna()]
        tm = vv["tight"]
        ft = vv.loc[tm, col]
        fn = vv.loc[~tm, col]
        by_k[str(kk)] = {
            "pooled_n": int(len(vv)),
            "P_tight": float(tm.mean()),
            "mean_fwd_u_tight": float(ft.mean()) if len(ft) else None,
            "mean_fwd_u_loose": float(fn.mean()) if len(fn) else None,
            "diff_tight_minus_loose": float(ft.mean() - fn.mean()) if len(ft) and len(fn) else None,
        }
    (OUT / "r4_phase3_gate_fwd_u_by_k_pooled.json").write_text(json.dumps(by_k, indent=2), encoding="utf-8")
    # keep legacy k20 file for downstream readers
    (OUT / "r4_phase3_gate_fwd_u_k20_pooled.json").write_text(
        json.dumps(by_k["20"], indent=2), encoding="utf-8"
    )

    # per day for K=20 (same as before)
    byd = []
    for d, g in pv.groupby("day"):
        tm = g["tight"]
        a = g.loc[tm, "fwd_u_k"]
        b = g.loc[~tm, "fwd_u_k"]
        byd.append(
            {
                "day": int(d),
                "mean_tight": float(a.mean()) if len(a) else None,
                "mean_loose": float(b.mean()) if len(b) else None,
                "n_tight": int(len(a)),
                "n_loose": int(len(b)),
            }
        )
    pd.DataFrame(byd).to_csv(OUT / "r4_phase3_gate_fwd_u_k20_by_day.csv", index=False)

    byd_multi = []
    for d, g in panel.groupby("day"):
        for kk in (5, 20, 100):
            col = f"fwd_u_{kk}"
            gg = g[g[col].notna()]
            if len(gg) == 0:
                continue
            tm = gg["tight"]
            a = gg.loc[tm, col]
            b = gg.loc[~tm, col]
            byd_multi.append(
                {
                    "day": int(d),
                    "K": kk,
                    "mean_tight": float(a.mean()) if len(a) else None,
                    "mean_loose": float(b.mean()) if len(b) else None,
                    "n_tight": int(len(a)),
                    "n_loose": int(len(b)),
                }
            )
    pd.DataFrame(byd_multi).to_csv(OUT / "r4_phase3_gate_fwd_u_by_k_by_day.csv", index=False)

    # --- merge gate onto Phase-1 enriched trades ---
    enr = pd.read_csv(ENR)
    gcols = panel[["day", "timestamp", "tight", "s5200", "s5300"]]
    mrg = enr.merge(gcols, on=["day", "timestamp"], how="inner")
    mrg.to_csv(OUT / "r4_phase3_trades_with_gate.csv", index=False)

    def participant_gate_table(mrg_df: pd.DataFrame, aggressor_side: str, name_col: str, min_n: int = 30) -> list[dict]:
        rows: list[dict] = []
        for tight, gate_lab in [(True, "tight"), (False, "loose")]:
            sub = mrg_df.loc[mrg_df["tight"] == tight]
            sub = sub[sub["aggressor"] == aggressor_side]
            for nm, g in sub.groupby(name_col):
                h = g["mark_20_u"].dropna()
                if len(h) < min_n:
                    continue
                rows.append(
                    {
                        "aggressor": aggressor_side,
                        "name": str(nm),
                        "gate": gate_lab,
                        "n": int(len(h)),
                        "mean_mark_20_u": float(h.mean()),
                        "median_mark_20_u": float(h.median()),
                    }
                )
        return rows

    pg_rows = participant_gate_table(mrg, "buy", "buyer", min_n=30) + participant_gate_table(
        mrg, "sell", "seller", min_n=30
    )
    pd.DataFrame(pg_rows).sort_values(["aggressor", "gate", "n"], ascending=[True, True, False]).to_csv(
        OUT / "r4_phase3_participant_mark20u_by_gate.csv", index=False
    )

    sub_all = mrg[(mrg["buyer"] == "Mark 01") & (mrg["seller"] == "Mark 22")]
    rows_all = []
    for tight, lab in [(True, "tight"), (False, "loose")]:
        h = sub_all.loc[sub_all["tight"] == tight, "mark_20_u"].dropna()
        rows_all.append(
            {
                "gate": lab,
                "n": int(len(h)),
                "mean": float(h.mean()) if len(h) else None,
                "median": float(h.median()) if len(h) else None,
            }
        )
    pd.DataFrame(rows_all).to_csv(OUT / "r4_phase3_mark01_22_all_syms_mark20u_by_gate.csv", index=False)

    # Mark01->Mark22 VEV_5300: tight vs loose mark_20_u
    sub = mrg[(mrg["buyer"] == "Mark 01") & (mrg["seller"] == "Mark 22") & (mrg["symbol"] == "VEV_5300")]
    rows = []
    for d, g in sub.groupby("day"):
        for tight, lab in [(True, "tight"), (False, "loose")]:
            h = g[g["tight"] == tight]["mark_20_u"].dropna()
            if len(h) >= 3:
                rows.append(
                    {
                        "day": int(d),
                        "gate": lab,
                        "n": len(h),
                        "mean_mark_20_u": float(h.mean()),
                        "median_mark_20_u": float(h.median()),
                    }
                )
    pd.DataFrame(rows).to_csv(OUT / "r4_phase3_mark01_22_vev5300_mark20u_by_gate_day.csv", index=False)

    # Pooled 01->22 5300
    pooled = []
    for tight, lab in [(True, "tight"), (False, "loose")]:
        h = sub[sub["tight"] == tight]["mark_20_u"].dropna()
        pooled.append({"gate": lab, "n": len(h), "mean_mark_20_u": float(h.mean()) if len(h) else None})
    pd.DataFrame(pooled).to_csv(OUT / "r4_phase3_mark01_22_vev5300_mark20u_by_gate_pooled.csv", index=False)

    # Burst x gate x U mark: use burst definition from trades same ts count>=4
    tr_parts = []
    for d in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        tr_parts.append(t)
    tr = pd.concat(tr_parts, ignore_index=True)
    burst_n = tr.groupby(["day", "timestamp"]).size().rename("n_prints").reset_index()
    burst_n["burst"] = burst_n["n_prints"] >= 4
    m2 = mrg.merge(burst_n[["day", "timestamp", "burst"]], on=["day", "timestamp"], how="left")
    m2["burst"] = m2["burst"].fillna(False)

    bx = []
    for burst, blab in [(True, "burst"), (False, "no_burst")]:
        for tight, tlab in [(True, "tight_gate"), (False, "loose_gate")]:
            h = m2[(m2["burst"] == burst) & (m2["tight"] == tight)]["mark_20_u"].dropna()
            if len(h) >= 20:
                bx.append(
                    {
                        "burst": blab,
                        "tight_gate": tlab,
                        "n": len(h),
                        "mean_mark_20_u": float(h.mean()),
                    }
                )
    pd.DataFrame(bx).to_csv(OUT / "r4_phase3_burst_gate_mark20u_pooled.csv", index=False)

    # Compare Phase1 burst effect without gate: from enr merge burst flag
    burst_n2 = burst_n[["day", "timestamp", "burst"]]
    en2 = enr.merge(burst_n2, on=["day", "timestamp"], how="left")
    en2["burst"] = en2["burst"].fillna(False)
    a = en2[en2["burst"]]["mark_20_u"].dropna()
    b = en2[~en2["burst"]]["mark_20_u"].dropna()
    cmp = {
        "phase1_style_burst_mean_u20": float(a.mean()) if len(a) else None,
        "phase1_style_non_mean_u20": float(b.mean()) if len(b) else None,
        "tight_subsample_mean_u20": float(m2[m2["tight"]]["mark_20_u"].dropna().mean()),
        "tight_and_burst_mean_u20": float(m2[m2["tight"] & m2["burst"]]["mark_20_u"].dropna().mean()),
    }
    (OUT / "r4_phase3_compare_phase1_burst_vs_gate.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")

    # inclineGod panels (scatter; hexbin can fail on some backends)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.scatter(panel["s5200"], panel["s5300"], s=1, alpha=0.15, c="0.3")
    ax.axvline(TH, color="r", ls="--", alpha=0.6)
    ax.axhline(TH, color="r", ls="--", alpha=0.6)
    ax.set_xlabel("VEV_5200 spread")
    ax.set_ylabel("VEV_5300 spread")
    ax.set_title("R4 spread scatter (pooled days)")
    ax = axes[1]
    ax.scatter(panel["s5200"], panel["s_u"], s=1, alpha=0.12, c="0.2")
    ax.set_xlabel("VEV_5200 spread")
    ax.set_ylabel("Extract spread")
    ax.set_title("5200 spread vs extract spread")
    fig.tight_layout()
    fig.savefig(OUT / "r4_phase3_spread_spread_panels.png", dpi=140)
    plt.close(fig)

    print("wrote phase3 outputs to", OUT)


if __name__ == "__main__":
    main()
