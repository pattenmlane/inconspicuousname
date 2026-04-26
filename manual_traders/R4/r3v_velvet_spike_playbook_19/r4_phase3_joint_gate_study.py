#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 L1 spread <= 2 same timestamp)
on Prosperity4Data/ROUND_4, matching round3work/vouchers_final_strategy inner-join logic.

Re-runs key Phase 1/2 summaries *conditional on joint_tight* at trade timestamp, plus
inclineGod-style spread–spread and spread vs extract dynamics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
TH = 2
K = 20
DAYS = (1, 2, 3)


def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    return df


def one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    v = v.assign(spread=(ask - bid).astype(float), mid=mid)
    return v[["timestamp", "spread", "mid"]].copy()


def aligned_panel(day: int) -> pd.DataFrame:
    df = load_prices(day)
    a = one_product(df, "VEV_5200").rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_product(df, "VEV_5300").rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_product(df, "VELVETFRUIT_EXTRACT").rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["day"] = day
    m["joint_tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["fwd_k"] = m["m_ext"].shift(-K) - m["m_ext"]
    m["d_ext"] = m["m_ext"].diff()
    return m


def load_trades(day: int) -> pd.DataFrame:
    t = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
    t["day"] = day
    t["product"] = t["symbol"].astype(str)
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = [aligned_panel(d) for d in DAYS]
    pan = pd.concat(panels, ignore_index=True)

    # --- Sonic: tight vs loose extract fwd (Phase 1 style on R4) ---
    rows = []
    for d in DAYS:
        g = pan[pan["day"] == d]
        t_arr = g.loc[g["joint_tight"], "fwd_k"].dropna()
        n_arr = g.loc[~g["joint_tight"], "fwd_k"].dropna()
        rows.append(
            {
                "day": d,
                "n_tight": int(len(t_arr)),
                "mean_fwd_tight": float(t_arr.mean()) if len(t_arr) else float("nan"),
                "mean_fwd_loose": float(n_arr.mean()) if len(n_arr) else float("nan"),
                "P_joint_tight": float(g["joint_tight"].mean()) if len(g) else 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "r4_phase3_extract_fwd20_tight_vs_loose_by_day.csv", index=False)

    # inclineGod: spread–spread
    ss = []
    for d in DAYS:
        g = pan[pan["day"] == d]
        sub = g[["s5200", "s5300"]].dropna()
        if len(sub) > 10:
            ss.append({"day": d, "corr_s5200_s5300": float(sub["s5200"].corr(sub["s5300"]))})
    pd.DataFrame(ss).to_csv(OUT / "r4_phase3_spread_spread_corr_by_day.csv", index=False)

    # spread vs "price" dynamics: |d_ext| vs s_ext
    dyn = []
    for d in DAYS:
        g = pan[pan["day"] == d].dropna(subset=["d_ext", "s_ext"])
        g = g[np.isfinite(g["d_ext"]) & np.isfinite(g["s_ext"])]
        if len(g) > 50:
            dyn.append(
                {
                    "day": d,
                    "corr_s_ext_abs_d_ext": float(g["s_ext"].corr(g["d_ext"].abs())),
                }
            )
    pd.DataFrame(dyn).to_csv(OUT / "r4_phase3_extract_spread_vs_abs_mid_change.csv", index=False)

    # Merge trades with joint_tight at timestamp
    trades = pd.concat([load_trades(d) for d in DAYS], ignore_index=True)
    gate = pan[["day", "timestamp", "joint_tight", "s5200", "s5300", "s_ext"]].copy()
    mt = trades.merge(gate, on=["day", "timestamp"], how="inner")
    mt["price"] = pd.to_numeric(mt["price"], errors="coerce")

    burst_n = (
        mt.groupby(["day", "timestamp"])
        .size()
        .reset_index(name="n_tick")
    )
    mt = mt.merge(burst_n, on=["day", "timestamp"], how="left")
    mt["is_burst"] = mt["n_tick"] >= 3

    # Forward extract +20 rows at trade ts (from panel)
    fwd_map = pan.set_index(["day", "timestamp"])["fwd_k"].to_dict()

    def fwd_ex(r):
        return fwd_map.get((int(r["day"]), int(r["timestamp"])), float("nan"))

    mt["fwd20_ex"] = mt.apply(fwd_ex, axis=1)

    mburst = mt[
        mt["is_burst"]
        & (mt["buyer"] == "Mark 01")
        & (mt["seller"] == "Mark 22")
        & mt["product"].str.startswith("VEV_")
    ]
    summ = (
        mburst.groupby(["day", "joint_tight"])["fwd20_ex"]
        .agg(n="count", mean="mean", median="median")
        .reset_index()
    )
    summ.to_csv(OUT / "r4_phase3_burst_m01_m22_vev_fwd20_by_gate.csv", index=False)

    pooled = (
        mburst.groupby("joint_tight")["fwd20_ex"]
        .agg(n="count", mean="mean")
        .reset_index()
    )
    (OUT / "r4_phase3_burst_m01_m22_fwd20_pooled_by_gate.json").write_text(
        json.dumps(pooled.to_dict("records"), indent=2)
    )

    _plots(pan)
    print("Phase 3 outputs ->", OUT)


def _plots(pan: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    s = pan.sample(min(8000, len(pan)), random_state=0) if len(pan) > 8000 else pan
    ax.scatter(s["s5200"], s["s5300"], alpha=0.15, s=4)
    ax.axhline(TH, color="r", linewidth=0.8)
    ax.axvline(TH, color="r", linewidth=0.8)
    ax.set_xlabel("VEV_5200 L1 spread")
    ax.set_ylabel("VEV_5300 L1 spread")
    ax.set_title("R4 spread–spread (joint tight = lower-left box)")
    fig.tight_layout()
    fig.savefig(OUT / "r4_phase3_spread_spread_scatter.png", dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for d in DAYS:
        g = pan[pan["day"] == d]["fwd_k"].dropna()
        ax2.hist(g, bins=40, alpha=0.35, label=f"day{d} all")
    ax2.set_title("Extract fwd20 (all timestamps, R4)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT / "r4_phase3_extract_fwd20_hist_by_day.png", dpi=120)
    plt.close(fig2)


if __name__ == "__main__":
    main()
