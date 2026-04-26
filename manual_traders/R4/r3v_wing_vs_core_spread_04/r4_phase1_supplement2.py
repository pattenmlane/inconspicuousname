#!/usr/bin/env python3
"""
Phase 1 supplement 2 — coverage gaps in suggested-direction text:

- **Per-day stability** for headline participant cells (from participant_markout_by_day_nosession.csv)
- **All distinct names** in trade tape + count of prints (as buyer, seller, any)
- **Reciprocity** on top directed pairs (A->B vs B->A; strength = min/max count)
- **Burst orchestration** roll-up (count bursts by (buyer_mode, seller_mode), mean n_sym)
- **Second-order** prints: **positive pooled mean residual** vs (buyer,seller,symbol,regime) baseline at k=20

Prerequisites: r4_phase1_counterparty_analysis.py has been run (outputs/phase1/*.csv).

  python3 manual_traders/R4/r3v_wing_vs_core_spread_04/r4_phase1_supplement2.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
DAYS = [1, 2, 3]
K_FOCUS = 20
MIN_DAILY = 20


def t_stat_1d(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    m = float(x.mean())
    s = float(x.std(ddof=1))
    if s < 1e-12:
        return float("nan")
    return m / (s / math.sqrt(len(x)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    byd = pd.read_csv(OUT / "participant_markout_by_day_nosession.csv")
    # -- Headline: Mark 67 buy_aggr extract tight non-burst k=20
    mask = (
        (byd["name"] == "Mark 67")
        & (byd["role"] == "buy_aggr")
        & (byd["symbol"] == "VELVETFRUIT_EXTRACT")
        & (byd["k"] == K_FOCUS)
        & (byd["spread_regime"] == "tight")
        & (byd["burst"] == 0)
    )
    st = byd.loc[mask, ["tape_day", "n", "mean", "t_stat", "frac_pos"]].copy()
    st = st.sort_values("tape_day")
    st["pass_n_ge_20"] = st["n"] >= MIN_DAILY
    st.to_csv(OUT / f"stability_m67_buy_aggr_extract_tight_k{K_FOCUS}_by_day.csv", index=False)
    summary = {
        "cell": "Mark 67, buy_aggr, VELVETFRUIT_EXTRACT, k=20, spread_regime=tight, burst=0",
        "min_daily_n": int(st["n"].min()) if len(st) else 0,
        "days_all_positive_mean": bool(len(st) and (st["mean"] > 0).all()) if len(st) else False,
        "per_day": st.to_dict(orient="records"),
    }
    (OUT / f"stability_m67_buy_aggr_extract_tight_k{K_FOCUS}_by_day.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # -- All names on tape
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            t = pd.read_csv(p, sep=";")
            t["tape_day"] = d
            frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    all_names = sorted(set(tr["buyer"].astype(str)) | set(tr["seller"].astype(str)))
    name_rows = []
    for nm in all_names:
        b = tr["buyer"] == nm
        s = tr["seller"] == nm
        name_rows.append(
            {
                "name": nm,
                "n_prints_as_buyer": int(b.sum()),
                "n_prints_as_seller": int(s.sum()),
                "n_prints_any_side": int((b | s).sum()),
            }
        )
    name_df = pd.DataFrame(name_rows).sort_values("n_prints_any_side", ascending=False)
    name_df.to_csv(OUT / "all_participant_names_print_counts.csv", index=False)
    (OUT / "all_participant_names_list.txt").write_text(
        "\n".join(all_names) + "\n", encoding="utf-8"
    )

    # -- Reciprocity
    edges = pd.read_csv(OUT / "graph_edges.csv")
    rows_r = []
    for _, r in edges.head(50).iterrows():
        a, b = str(r["buyer"]), str(r["seller"])
        fwd = int(r["count"])
        rev = edges.loc[(edges["buyer"] == b) & (edges["seller"] == a), "count"]
        rev_c = int(rev.iloc[0]) if len(rev) else 0
        mx = max(fwd, rev_c, 1)
        rec = min(fwd, rev_c) / mx
        rows_r.append(
            {
                "A": a,
                "B": b,
                "A_to_B_count": fwd,
                "B_to_A_count": rev_c,
                "reciprocity_min_max": rec,
            }
        )
    pd.DataFrame(rows_r).to_csv(OUT / "graph_reciprocity_top50_forward_edges.csv", index=False)

    # -- Burst orchestration roll-up
    be = pd.read_csv(OUT / "burst_events.csv")
    orch = (
        be.groupby(["buyer_mode", "seller_mode"], dropna=False)
        .agg(
            n_burst_timestamps=("timestamp", "count"),
            mean_n_prints=("n_prints", "mean"),
            mean_n_sym=("n_sym", "mean"),
        )
        .reset_index()
        .sort_values("n_burst_timestamps", ascending=False)
    )
    orch.to_csv(OUT / "burst_orchestrator_summary_by_mode.csv", index=False)

    # -- Second-order vs **coarse** (buyer,symbol,k) baseline: spread regime adds signal
    #     not captured by (buyer,seller,symbol,regime) full cell means (per-print
    #     residuals in per_print_with_baseline sum to 0 by construction within each
    #     (buyer,seller,symbol,regime,k) group).
    pp = pd.read_csv(OUT / "per_print_with_baseline.csv")
    p20 = pp[pp["k"] == K_FOCUS]
    for col in ("buyer", "seller", "symbol", "spread_regime"):
        p20 = p20[p20[col].notna()]
    coarse = (
        p20.groupby(["buyer", "seller", "symbol"], as_index=False, dropna=False)
        .agg(baseline_fwd_pooled_k20=("fwd_same", "mean"), n_pooled=("fwd_same", "size"))
    )
    m = p20.merge(coarse, on=["buyer", "seller", "symbol"], how="left")
    m["residual_vs_pair_symbol_pool"] = m["fwd_same"] - m["baseline_fwd_pooled_k20"]
    gcols = ["buyer", "seller", "symbol", "spread_regime"]
    so_rows = []
    for key, g in m.groupby(gcols, dropna=False):
        res = g["residual_vs_pair_symbol_pool"].to_numpy(dtype=float)
        res = res[np.isfinite(res)]
        n = len(res)
        if n < 25:
            continue
        mean_r = float(np.mean(res))
        t = t_stat_1d(res)
        d = dict(zip(gcols, key, strict=True))
        d["k"] = K_FOCUS
        d["n"] = n
        d["mean_residual_vs_pair_pooled_baseline"] = mean_r
        d["t_stat"] = t
        d["frac_pos"] = float(np.mean(res > 0))
        byday = g.groupby("tape_day")["residual_vs_pair_symbol_pool"].mean()
        d["n_days"] = int(byday.shape[0])
        d["all_3_days_same_sign_as_pooled"] = bool(
            len(byday) == 3
            and (mean_r > 0 and (byday > 0).all() or mean_r < 0 and (byday < 0).all())
        )
        so_rows.append(d)
    so = pd.DataFrame(so_rows)
    if len(so):
        so["abs_t"] = so["t_stat"].abs()
        so = so.sort_values("abs_t", ascending=False, na_position="last").drop(
            columns=["abs_t"], errors="ignore"
        )
    out_so = OUT / f"second_order_regime_vs_pair_pooled_baseline_k{K_FOCUS}.csv"
    so.to_csv(out_so, index=False)
    top = so[so["t_stat"] > 0].head(20) if len(so) else so
    (OUT / f"second_order_regime_vs_pair_pooled_baseline_k{K_FOCUS}_top_positive_t.json").write_text(
        json.dumps(top.to_dict(orient="records") if len(top) else [], indent=2), encoding="utf-8"
    )

    # Index update (append once; avoid duplicate on re-run)
    marker = f"--- supplement2 ({Path(__file__).name}) ---"
    add = f"""

{marker}

stability: stability_m67_buy_aggr_extract_tight_k{K_FOCUS}_by_day.csv, .json
all names: all_participant_names_print_counts.csv, all_participant_names_list.txt
reciprocity: graph_reciprocity_top50_forward_edges.csv
orchestrator: burst_orchestrator_summary_by_mode.csv
second-order: second_order_regime_vs_pair_pooled_baseline_k{K_FOCUS}.csv (regime effect vs (buyer,seller,symbol) pool)
"""
    index_path = OUT / "phase1_bullet_file_index.txt"
    if index_path.is_file() and marker not in index_path.read_text(encoding="utf-8"):
        index_path.write_text(index_path.read_text(encoding="utf-8") + add, encoding="utf-8")

    print("Wrote supplement2 files to", OUT)


if __name__ == "__main__":
    main()
