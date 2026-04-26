#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic **joint gate** on tape (same convention as
``round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py``):
inner-join **VEV_5200**, **VEV_5300**, **VELVETFRUIT_EXTRACT** on ``timestamp``;
``tight`` = (s5200 <= TH) & (s5300 <= TH); forward extract mid K=20 bars.

Also: inclineGod-style **spread–spread** / **spread vs mid** tables; merge gate onto
Phase-1-style enriched trades for **Mark × gate** markouts and burst×gate splits.

Run from repo root:
  python3 manual_traders/R4/r4_phase1_marks/analyze_phase3.py
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FILE = Path(__file__).resolve()
REPO = FILE.parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = FILE.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
K = 20
VEV_5200, VEV_5300, EXTRACT = "VEV_5200", "VEV_5300", "VELVETFRUIT_EXTRACT"


def load_p1():
    spec = importlib.util.spec_from_file_location("p1", FILE.parent / "analyze_phase1.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def _one_product_r4(df: pd.DataFrame, product: str) -> pd.DataFrame:
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


def aligned_panel_r4(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    a = _one_product_r4(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = _one_product_r4(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = _one_product_r4(df, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e[["timestamp", "m_ext", "s_ext"]], on="timestamp", how="inner"
    )
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["day"] = day
    return m


def add_forward_and_tight(m: pd.DataFrame, *, th: int = TH, k: int = K) -> pd.DataFrame:
    out = m.copy()
    out["tight"] = (out["s5200"] <= th) & (out["s5300"] <= th)
    out["m_ext_f"] = out["m_ext"].shift(-k)
    out["fwd_k"] = out["m_ext_f"] - out["m_ext"]
    return out


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"),) * 4
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return (float(np.mean(a)), float(np.mean(b)), float(r.statistic), float(r.pvalue))


def joint_gate_summary() -> None:
    lines = ["Round 4 joint gate (inner join 5200+5300+extract, TH=%d, K=%d)\n" % (TH, K)]
    for day in DAYS:
        p = add_forward_and_tight(aligned_panel_r4(day))
        ok = p["fwd_k"].notna()
        pv = p.loc[ok]
        t_mask = pv["tight"]
        ft = pv.loc[t_mask, "fwd_k"].values
        fn = pv.loc[~t_mask, "fwd_k"].values
        mt, mn, tstat, pval = welch_t(ft, fn)
        p_tight = float(t_mask.mean()) if len(pv) else float("nan")
        lines.append(
            f"day {day}: n_valid_fwd={len(pv)} P(tight)={p_tight:.4f}\n"
            f"  mean(fwd)|tight={mt:.6g} mean(fwd)|not={mn:.6g} Welch t={tstat:.4f} p={pval:.4g}\n"
            f"  corr(s5200,s5300)={pv['s5200'].corr(pv['s5300']):.4f} "
            f"corr(s5200,m_ext)={pv['s5200'].corr(pv['m_ext']):.4f} "
            f"corr(s5300,m_ext)={pv['s5300'].corr(pv['m_ext']):.4f}\n"
        )
    (OUT / "phase3_joint_gate_summary_r4.txt").write_text("".join(lines), encoding="utf-8")


def spread_correlation_matrix() -> None:
    """inclineGod: spread–spread and spread vs mid on aligned rows (all days pooled)."""
    rows = []
    for day in DAYS:
        p = add_forward_and_tight(aligned_panel_r4(day))
        rows.append(p)
    allp = pd.concat(rows, ignore_index=True)
    cols = ["s5200", "s5300", "s_ext", "mid5200", "mid5300", "m_ext"]
    sub = allp[cols].dropna()
    C = sub.corr()
    C.to_csv(OUT / "phase3_spread_mid_correlation_matrix_r4.csv")
    # spread–spread block
    ss = allp[["s5200", "s5300", "s_ext"]].dropna()
    ss.corr().to_csv(OUT / "phase3_spread_spread_only_r4.csv")


def inclineGod_panels_png() -> None:
    """One figure: scatter s5200 vs s5300 colored by tight; spreads vs time head; m_ext vs s_ext."""
    day = 1
    p = add_forward_and_tight(aligned_panel_r4(day))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    u_ts = p["timestamp"].unique()[:800]
    ph = p[p["timestamp"].isin(u_ts)]
    ax = axes[0, 0]
    ax.plot(ph["timestamp"], ph["s5200"], lw=0.7, label="s5200")
    ax.plot(ph["timestamp"], ph["s5300"], lw=0.7, label="s5300")
    ax.axhline(TH, color="r", ls="--", lw=0.8)
    ax.set_title(f"R4 day {day}: spreads (first 800 ts)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    sc = ax.scatter(p["s5200"], p["s5300"], c=p["tight"].astype(int), s=3, alpha=0.35, cmap="coolwarm")
    ax.axvline(TH, color="k", ls=":", lw=0.6)
    ax.axhline(TH, color="k", ls=":", lw=0.6)
    ax.set_xlabel("s5200")
    ax.set_ylabel("s5300")
    ax.set_title("Spread–spread (colored: tight=1)")

    ax = axes[1, 0]
    ax.scatter(p["s_ext"], p["m_ext"], c=p["tight"].astype(int), s=3, alpha=0.35, cmap="coolwarm")
    ax.set_xlabel("extract spread ticks")
    ax.set_ylabel("extract mid")
    ax.set_title("Spread vs price (extract)")

    ax = axes[1, 1]
    ok = p["fwd_k"].notna()
    ax.hist(p.loc[ok & p["tight"], "fwd_k"], bins=40, alpha=0.6, label="tight", density=True)
    ax.hist(p.loc[ok & ~p["tight"], "fwd_k"], bins=40, alpha=0.6, label="not tight", density=True)
    ax.set_title(f"fwd extract Δ (K={K})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "phase3_r4_inclineGod_panels.png", dpi=120)
    plt.close(fig)


def merge_trades_with_tight(p1) -> pd.DataFrame:
    te = p1.build_trade_enriched()
    parts = []
    for day in DAYS:
        p = add_forward_and_tight(aligned_panel_r4(day))[
            ["day", "timestamp", "tight", "s5200", "s5300", "s_ext", "fwd_k"]
        ]
        sub = te[te["day"] == day].merge(
            p.rename(columns={"fwd_k": "fwd_extract_gate_panel_K"}),
            on=["day", "timestamp"],
            how="left",
        )
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def mark_gate_x_pair(m: pd.DataFrame) -> None:
    """Primary pairs only (Phase 2 focus)."""
    pairs = {("Mark 01", "Mark 22"), ("Mark 14", "Mark 38"), ("Mark 55", "Mark 01")}
    rows = []
    for (b, s, sym), g in m.groupby(["buyer", "seller", "symbol"]):
        if (b, s) not in pairs or len(g) < 15:
            continue
        for tight_flag, name in [(True, "tight"), (False, "loose")]:
            sub = g[g["tight"] == tight_flag]
            x = sub["fwd_same_20"].astype(float).dropna().values
            if len(x) < 6:
                continue
            rows.append(
                {
                    "buyer": b,
                    "seller": s,
                    "symbol": sym,
                    "gate": name,
                    "n": len(x),
                    "mean_fwd20": float(np.mean(x)),
                    "frac_pos": float(np.mean(x > 0)),
                }
            )
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "phase3_mark_pair_symbol_gate_markout.csv", index=False)


def gate_x_burst_extract(m: pd.DataFrame, tr: pd.DataFrame) -> None:
    cnt = tr.groupby(["day", "timestamp"]).size().reset_index(name="burst_n")
    x = m.merge(cnt, on=["day", "timestamp"], how="left")
    x["burst_ge4"] = x["burst_n"] >= 4
    rows = []
    for tight in (True, False):
        for burst in (True, False):
            g = x[(x["tight"] == tight) & (x["burst_ge4"] == burst)]
            v = g["fwd_EXTRACT_20"].astype(float).dropna().values
            if len(v) < 1:
                continue
            rows.append(
                {
                    "joint_tight": tight,
                    "burst_ge4": burst,
                    "n": len(v),
                    "mean_fwd_EXTRACT_20": float(np.mean(v)),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "phase3_gate_x_burst_extract_fwd20.csv", index=False)


def compare_phase1_burst_without_gate() -> None:
    """Text note: Phase 1 pooled burst vs non on same metric for reference."""
    lines = ["Compare to Phase 1 `burst_extract_welch_ge4_vs_lt4.txt` (no gate).\n"]
    lines.append("Phase 3 adds stratification by `tight` in phase3_gate_x_burst_extract_fwd20.csv.\n")
    (OUT / "phase3_vs_phase1_burst_note.txt").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    p1 = load_p1()
    joint_gate_summary()
    spread_correlation_matrix()
    inclineGod_panels_png()
    m = merge_trades_with_tight(p1)
    slim = m[
        [
            "day",
            "timestamp",
            "buyer",
            "seller",
            "symbol",
            "tight",
            "s5200",
            "s5300",
            "fwd_same_20",
            "fwd_EXTRACT_20",
        ]
    ]
    slim.head(5000).to_csv(OUT / "phase3_trades_gate_head5000.csv", index=False)

    tr = pd.concat(
        [pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS],
        ignore_index=True,
    )
    mark_gate_x_pair(m)
    gate_x_burst_extract(m, tr)
    compare_phase1_burst_without_gate()
    print("Phase 3 done ->", OUT)


if __name__ == "__main__":
    main()
