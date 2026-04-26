#!/usr/bin/env python3
"""
Round 4 Phase 3 — Sonic joint gate (VEV_5200 & VEV_5300 L1 spread <= 2 same timestamp),
spread–spread / spread–price panels, and counterparty × gate interactions.

Convention matches round3work/vouchers_final_strategy/analyze_vev_5200_5300_tight_gate_r3.py:
  inner join timestamps where 5200, 5300, extract each have a row; dedupe per product per ts;
  spread = ask_price_1 - bid_price_1; tight = (s5200 <= TH) & (s5300 <= TH).

Trade merge: each trade (day, timestamp, symbol) joined to panel row on same (day, timestamp)
for gate flags and panel mids/spreads; forward mid on traded symbol = K rows ahead in
full long prices table (same as Phase 1).

Outputs: manual_traders/R4/r3v_wide_book_passive_11/analysis_outputs_r4_phase3/

Run: python3 manual_traders/R4/r3v_wide_book_passive_11/r4_phase3_sonic_gate.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs_r4_phase3"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
K = 20
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"


def one_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")
    )
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


def aligned_panel_day(day: int) -> pd.DataFrame:
    p = DATA / f"prices_round_4_day_{day}.csv"
    df = pd.read_csv(p, sep=";")
    df = df[df["day"] == day] if "day" in df.columns else df
    a = one_product(df, VEV_5200).rename(columns={"spread": "s5200", "mid": "mid5200"})
    b = one_product(df, VEV_5300).rename(columns={"spread": "s5300", "mid": "mid5300"})
    e = one_product(df, EXTRACT).rename(columns={"spread": "s_ext", "mid": "m_ext"})
    m = a.merge(b, on="timestamp", how="inner").merge(
        e, on="timestamp", how="inner"
    )
    m["day"] = day
    m = m.sort_values("timestamp").reset_index(drop=True)
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    m["m_ext_f"] = m["m_ext"].shift(-K)
    m["fwd_k_extract"] = m["m_ext_f"] - m["m_ext"]
    return m


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (float("nan"),) * 4
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(a.mean()), float(b.mean()), float(r.statistic), float(r.pvalue)


def load_prices_long() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    pr = pd.concat(frames, ignore_index=True)
    pr["mid"] = pd.to_numeric(pr["mid_price"], errors="coerce")
    pr["bid1"] = pd.to_numeric(pr["bid_price_1"], errors="coerce")
    pr["ask1"] = pd.to_numeric(pr["ask_price_1"], errors="coerce")
    pr["spread"] = pr["ask1"] - pr["bid1"]
    return pr


def add_fwd(pr: pd.DataFrame, Kk: int) -> pd.DataFrame:
    pr = pr.sort_values(["day", "product", "timestamp"])
    col = f"fwd_mid_{Kk}"
    pr[col] = pr.groupby(["day", "product"], group_keys=False)["mid"].transform(
        lambda s: s.shift(-Kk) - s
    )
    return pr


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    for c in ("symbol", "buyer", "seller"):
        tr[c] = tr[c].astype(str)
    return tr


def main() -> None:
    # --- Panel: tight vs loose extract forward (R3 replication on R4) ---
    rows_panel = []
    rows_corr = []
    for d in DAYS:
        pan = aligned_panel_day(d)
        v = pan["fwd_k_extract"].notna()
        pv = pan.loc[v]
        t_mask = pv["tight"]
        mt, mn, tstat, pval = welch_t(
            pv.loc[t_mask, "fwd_k_extract"].values,
            pv.loc[~t_mask, "fwd_k_extract"].values,
        )
        rows_panel.append(
            {
                "day": d,
                "n_tight": int(t_mask.sum()),
                "n_loose": int((~t_mask).sum()),
                "mean_fwd_extract_tight": mt,
                "mean_fwd_extract_loose": mn,
                "welch_t": tstat,
                "p_value": pval,
                "P_tight": float(t_mask.mean()),
            }
        )
        rows_corr.append(
            {
                "day": d,
                "corr_s5200_s5300": float(pv["s5200"].corr(pv["s5300"])),
                "corr_s5200_m_ext": float(pv["s5200"].corr(pv["m_ext"])),
                "corr_s5300_m_ext": float(pv["s5300"].corr(pv["m_ext"])),
                "corr_s5200_fwd_k": float(pv["s5200"].corr(pv["fwd_k_extract"])),
                "corr_s5300_fwd_k": float(pv["s5300"].corr(pv["fwd_k_extract"])),
                "corr_s_ext_fwd_k": float(pv["s_ext"].corr(pv["fwd_k_extract"])),
            }
        )

    pd.DataFrame(rows_panel).to_csv(OUT / "p3_01_extract_fwdK_tight_vs_loose_by_day.csv", index=False)
    pd.DataFrame(rows_corr).to_csv(OUT / "p3_02_spread_correlations_panel_by_day.csv", index=False)

    # Pooled panel (all days concat) — one Welch on pooled valid rows
    all_p = pd.concat([aligned_panel_day(d) for d in DAYS], ignore_index=True)
    pv = all_p[all_p["fwd_k_extract"].notna()]
    t_mask = pv["tight"]
    mt, mn, tstat, pval = welch_t(
        pv.loc[t_mask, "fwd_k_extract"].values,
        pv.loc[~t_mask, "fwd_k_extract"].values,
    )
    Path(OUT / "p3_00_pooled_extract_fwd_tight_vs_loose.txt").write_text(
        f"Pooled days {DAYS}: n_tight={int(t_mask.sum())} n_loose={int((~t_mask).sum())}\n"
        f"mean_fwd_extract_K{K} tight={mt:.6g} loose={mn:.6g} welch_t={tstat:.4f} p={pval:.4g}\n"
        f"P(tight)={float(t_mask.mean()):.4f}\n"
        f"corr(s5200,s5300)={float(pv['s5200'].corr(pv['s5300'])):.4f}\n"
    )

    # --- Trades merged to panel (gate at trade timestamp) ---
    pr = add_fwd(load_prices_long(), K)
    px = pr.rename(columns={"product": "symbol"})
    sym_cols = ["day", "timestamp", "symbol", "mid", "spread", f"fwd_mid_{K}"]
    tr = load_trades()
    m = tr.merge(px[sym_cols], on=["day", "timestamp", "symbol"], how="left")

    panels = pd.concat(
        [aligned_panel_day(d)[["day", "timestamp", "tight", "s5200", "s5300", "s_ext"]].rename(
            columns={"tight": "sonic_tight_panel", "s_ext": "s_ext_panel"}
        ) for d in DAYS],
        ignore_index=True,
    )
    m = m.merge(panels, on=["day", "timestamp"], how="left")
    m["sonic_tight"] = m["sonic_tight_panel"].fillna(False)

    # Three-way: (buyer, seller, symbol, sonic_tight) — min n per cell
    g = (
        m.groupby(["buyer", "seller", "symbol", "sonic_tight"])[f"fwd_mid_{K}"]
        .agg(["count", "mean"])
        .reset_index()
    )
    g = g[g["count"] >= 15]
    g.to_csv(OUT / "p3_03_three_way_pair_symbol_gate_fwd20.csv", index=False)

    # Pivot: Mark 01 -> 22, key symbols
    key = m[(m["buyer"] == "Mark 01") & (m["seller"] == "Mark 22")]
    pivot_rows = []
    for sym in key["symbol"].unique():
        sub = key[key["symbol"] == sym]
        for tight in (True, False):
            s = sub[sub["sonic_tight"] == tight]
            x = pd.to_numeric(s[f"fwd_mid_{K}"], errors="coerce").dropna()
            if len(x) < 10:
                continue
            pivot_rows.append(
                {
                    "symbol": sym,
                    "sonic_tight": tight,
                    "n": len(x),
                    "mean_fwd20": float(x.mean()),
                    "frac_pos": float((x > 0).mean()),
                }
            )
    pd.DataFrame(pivot_rows).sort_values(["symbol", "sonic_tight"]).to_csv(
        OUT / "p3_04_mark01_22_fwd20_by_symbol_and_gate.csv", index=False
    )

    # Mark 67, Mark 55->14 extract: by gate + by day
    tw = []
    for mk_filter in (
        ("Mark 67", None, None),
        ("Mark 55", "Mark 14", EXTRACT),
    ):
        if mk_filter[1] is None:
            sub = m[(m["buyer"] == mk_filter[0]) | (m["seller"] == mk_filter[0])]
            label = "Mark67_any"
        else:
            sub = m[
                (m["buyer"] == mk_filter[0])
                & (m["seller"] == mk_filter[1])
                & (m["symbol"] == mk_filter[2])
            ]
            label = "Mark55_Mark14_extract"
        for d in DAYS:
            for tight in (True, False):
                s = sub[(sub["day"] == d) & (sub["sonic_tight"] == tight)]
                x = pd.to_numeric(s[f"fwd_mid_{K}"], errors="coerce").dropna()
                tw.append(
                    {
                        "cohort": label,
                        "day": d,
                        "sonic_tight": tight,
                        "n": len(x),
                        "mean_fwd20": float(x.mean()) if len(x) else float("nan"),
                    }
                )
    pd.DataFrame(tw).to_csv(OUT / "p3_05_key_marks_fwd20_by_day_and_gate.csv", index=False)

    # Compare Phase1 headline without gate: overall Mark01->22 mean vs tight-only
    o = m[(m["buyer"] == "Mark 01") & (m["seller"] == "Mark 22")]
    x_all = pd.to_numeric(o[f"fwd_mid_{K}"], errors="coerce").dropna()
    x_tight = pd.to_numeric(
        o.loc[o["sonic_tight"], f"fwd_mid_{K}"], errors="coerce"
    ).dropna()
    x_loose = pd.to_numeric(
        o.loc[~o["sonic_tight"], f"fwd_mid_{K}"], errors="coerce"
    ).dropna()
    Path(OUT / "p3_06_mark01_22_gate_interaction_summary.txt").write_text(
        f"Mark 01 -> Mark 22 all symbols pooled:\n"
        f"  n_all={len(x_all)} mean_fwd20={float(x_all.mean()):.6g}\n"
        f"  n_tight={len(x_tight)} mean={float(x_tight.mean()) if len(x_tight) else float('nan'):.6g}\n"
        f"  n_loose={len(x_loose)} mean={float(x_loose.mean()) if len(x_loose) else float('nan'):.6g}\n"
        f"Welch tight vs loose: {welch_t(x_tight.values, x_loose.values)}\n"
    )

    # inclineGod: spread-spread matrix pooled over panel rows
    pv_all = pd.concat([aligned_panel_day(d) for d in DAYS], ignore_index=True)
    pv_all = pv_all[pv_all["fwd_k_extract"].notna()]
    cols = ["s5200", "s5300", "s_ext"]
    corr_mat = pv_all[cols].corr()
    corr_mat.to_csv(OUT / "p3_07_spread_spread_matrix_pooled.csv")

    readme = []
    readme.append(Path(OUT / "p3_00_pooled_extract_fwd_tight_vs_loose.txt").read_text())
    readme.append("\n--- By day tight vs loose (extract fwd K=20) ---\n")
    readme.append(pd.read_csv(OUT / "p3_01_extract_fwdK_tight_vs_loose_by_day.csv").to_string(index=False))
    readme.append("\n\n--- Spread correlations by day ---\n")
    readme.append(pd.read_csv(OUT / "p3_02_spread_correlations_panel_by_day.csv").to_string(index=False))
    readme.append("\n\n--- Mark01->22 gate summary ---\n")
    readme.append(Path(OUT / "p3_06_mark01_22_gate_interaction_summary.txt").read_text())
    Path(OUT / "00_README_PHASE3.txt").write_text("".join(readme))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
