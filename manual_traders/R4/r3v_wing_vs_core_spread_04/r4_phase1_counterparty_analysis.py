#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned markouts (full bullets in round4work/suggested direction.txt).

Horizon K: **K steps** = K increments of the tape timestamp index (same convention as
vouchers_final_strategy: forward return at t+K means the mid at the K-th subsequent
timestamp row for that product on the same tape day).

Outputs under manual_traders/R4/r3v_wing_vs_core_spread_04/outputs/phase1/
  - participant_markout_by_day.csv
  - participant_markout_pooled.csv
  - pair_baseline_residuals.csv
  - graph_edges.csv, graph_top_pairs.txt
  - burst_events.csv, burst_forward_extract.csv
  - burst_forward_vev5200_vev5300_rows.csv, burst_forward_core_vev_summary.csv
  - passive_adverse_by_pair.csv
  - phase1_summary.json

Run from repo root:
  python3 manual_traders/R4/r3v_wing_vs_core_spread_04/r4_phase1_counterparty_analysis.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
HORIZONS = (5, 20, 100)
PRODUCTS_R4 = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def load_prices_day(day: int) -> pd.DataFrame:
    p = DATA / f"prices_round_4_day_{day}.csv"
    df = pd.read_csv(p, sep=";")
    df["day"] = int(day)
    return df


def build_mid_matrix(px: pd.DataFrame) -> tuple[np.ndarray, dict[str, int], dict[int, int]]:
    """Rows = timestamp index 0..T-1, cols = product. Returns mids, prod_index, ts_to_idx."""
    ts_sorted = sorted(px["timestamp"].unique())
    ts_to_idx = {int(t): i for i, t in enumerate(ts_sorted)}
    T = len(ts_sorted)
    prod_index = {p: i for i, p in enumerate(PRODUCTS_R4)}
    mids = np.full((T, len(PRODUCTS_R4)), np.nan, dtype=float)
    for prod in PRODUCTS_R4:
        sub = px.loc[px["product"] == prod, ["timestamp", "mid_price"]].drop_duplicates(
            "timestamp"
        )
        j = prod_index[prod]
        for _, r in sub.iterrows():
            ti = ts_to_idx.get(int(r["timestamp"]))
            if ti is not None:
                mids[ti, j] = float(r["mid_price"])
    return mids, prod_index, ts_to_idx


def forward_delta(mids: np.ndarray, ti: int, j: int, k: int) -> float:
    if ti + k >= mids.shape[0] or ti + k < 0:
        return float("nan")
    a, b = mids[ti, j], mids[ti + k, j]
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return b - a


def spread_at(px_row: pd.Series) -> float | None:
    try:
        b = float(px_row["bid_price_1"])
        a = float(px_row["ask_price_1"])
    except (TypeError, ValueError, KeyError):
        return None
    if pd.isna(px_row.get("bid_price_1")) or pd.isna(px_row.get("ask_price_1")):
        return None
    return a - b


def classify_aggression(price: float, bid1: float, ask1: float) -> str:
    if price >= ask1:
        return "buy_aggr"
    if price <= bid1:
        return "sell_aggr"
    return "inside"


def main() -> None:
    all_trades: list[pd.DataFrame] = []
    for d in DAYS:
        tpath = DATA / f"trades_round_4_day_{d}.csv"
        if not tpath.is_file():
            continue
        td = pd.read_csv(tpath, sep=";")
        td["tape_day"] = d
        all_trades.append(td)
    tr = pd.concat(all_trades, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)

    # --- Merge book at trade time for aggression + spread ---
    book_rows: list[pd.DataFrame] = []
    for d in DAYS:
        px = load_prices_day(d)
        keep = px[px["product"].isin(PRODUCTS_R4)].copy()
        keep = keep.rename(columns={"product": "symbol", "day": "tape_day"})
        book_rows.append(keep[["tape_day", "timestamp", "symbol", "bid_price_1", "ask_price_1", "mid_price"]])
    book = pd.concat(book_rows, ignore_index=True)

    m = tr.merge(
        book,
        on=["tape_day", "timestamp", "symbol"],
        how="left",
        suffixes=("", "_book"),
    )
    m["spread"] = m["ask_price_1"] - m["bid_price_1"]
    m["aggression"] = [
        classify_aggression(float(p), float(b), float(a))
        if pd.notna(p) and pd.notna(b) and pd.notna(a)
        else "unknown"
        for p, b, a in zip(m["price"], m["bid_price_1"], m["ask_price_1"], strict=True)
    ]

    # Per-(day,symbol) spread quantiles for regime tag
    def spread_regime_col(df: pd.DataFrame) -> pd.Series:
        out = []
        for (_, _), g in df.groupby(["tape_day", "symbol"], sort=False):
            s = g["spread"]
            lo, hi = s.quantile(0.33), s.quantile(0.66)
            lab = pd.Series("mid", index=g.index)
            lab.loc[s.isna()] = "unknown"
            lab.loc[s.notna() & (s <= lo)] = "tight"
            lab.loc[s.notna() & (s >= hi)] = "wide"
            out.append(lab)
        return pd.concat(out).sort_index()

    m["spread_regime"] = spread_regime_col(m)

    # Session quartile: timestamp percentile rank within tape day
    m["session_q"] = (
        m.groupby("tape_day")["timestamp"].rank(method="average", pct=True).mul(4).clip(0, 3.999).astype(int)
    )

    burst_size = m.groupby(["tape_day", "timestamp"]).size().rename("burst_n")
    m = m.merge(burst_size, on=["tape_day", "timestamp"], how="left")
    m["burst"] = (m["burst_n"] >= 4).astype(int)

    # --- Participant-level markouts ---
    names = sorted(set(m["buyer"].astype(str)) | set(m["seller"].astype(str)))
    rows_part: list[dict] = []

    for tape_day in DAYS:
        px = load_prices_day(tape_day)
        mids, prod_index, ts_to_idx = build_mid_matrix(px)
        T = mids.shape[0]
        sub = m[m["tape_day"] == tape_day]
        u_idx = prod_index["VELVETFRUIT_EXTRACT"]
        h_idx = prod_index["HYDROGEL_PACK"]

        for _, r in sub.iterrows():
            sym = str(r["symbol"])
            if sym not in prod_index:
                continue
            ti = ts_to_idx.get(int(r["timestamp"]))
            if ti is None:
                continue
            sj = prod_index[sym]
            for k in HORIZONS:
                d_same = forward_delta(mids, ti, sj, k)
                d_u = forward_delta(mids, ti, u_idx, k)
                d_h = forward_delta(mids, ti, h_idx, k)
                base = {
                    "tape_day": tape_day,
                    "name": "",
                    "role": "",
                    "symbol": sym,
                    "k": k,
                    "spread_regime": r["spread_regime"],
                    "session_q": int(r["session_q"]),
                    "burst": int(r["burst"]),
                    "fwd_same": d_same,
                    "fwd_extract": d_u,
                    "fwd_hydro": d_h,
                }
                buyer, seller = str(r["buyer"]), str(r["seller"])
                for nm, role in ((buyer, "buyer_any"), (seller, "seller_any")):
                    rr = dict(base)
                    rr["name"] = nm
                    rr["role"] = role
                    rows_part.append(rr)
                if r["aggression"] == "buy_aggr":
                    rr = dict(base)
                    rr["name"] = buyer
                    rr["role"] = "buy_aggr"
                    rows_part.append(rr)
                if r["aggression"] == "sell_aggr":
                    rr = dict(base)
                    rr["name"] = seller
                    rr["role"] = "sell_aggr"
                    rows_part.append(rr)

    part_df = pd.DataFrame(rows_part)
    part_df.to_csv(OUT / "participant_markout_long.csv", index=False)

    def agg_markout(g: pd.DataFrame) -> pd.Series:
        col = "fwd_same"
        x = g[col].dropna()
        if len(x) < 15:
            return pd.Series({"n": len(x), "mean": np.nan, "median": np.nan, "t_stat": np.nan, "frac_pos": np.nan})
        mean = float(x.mean())
        std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        t_stat = mean / (std / math.sqrt(len(x))) if std > 1e-12 else float("nan")
        return pd.Series(
            {
                "n": len(x),
                "mean": mean,
                "median": float(x.median()),
                "t_stat": t_stat,
                "frac_pos": float((x > 0).mean()),
            }
        )

    gcols = ["name", "role", "symbol", "k", "spread_regime", "session_q", "burst", "tape_day"]
    by_day = part_df.groupby(gcols, dropna=False).apply(agg_markout).reset_index()
    by_day.to_csv(OUT / "participant_markout_by_day.csv", index=False)

    gcols_pool = ["name", "role", "symbol", "k", "spread_regime", "burst"]
    pooled = part_df.groupby(gcols_pool, dropna=False).apply(agg_markout).reset_index()
    pooled.to_csv(OUT / "participant_markout_pooled.csv", index=False)

    # Same as by_day but **without** session_q (cleaner per-day n for stability tables)
    gcols_day = ["name", "role", "symbol", "k", "spread_regime", "burst", "tape_day"]
    by_day_nosession = part_df.groupby(gcols_day, dropna=False).apply(agg_markout).reset_index()
    by_day_nosession.to_csv(OUT / "participant_markout_by_day_nosession.csv", index=False)

    # --- Two-hop chains: consecutive tape prints (same day) where seller(t-1)==buyer(t) ---
    tr_s = tr.sort_values(["tape_day", "timestamp"]).reset_index(drop=True)
    tr_s["prev_buyer"] = tr_s.groupby("tape_day")["buyer"].shift(1)
    tr_s["prev_seller"] = tr_s.groupby("tape_day")["seller"].shift(1)
    tr_s["prev_ts"] = tr_s.groupby("tape_day")["timestamp"].shift(1)
    tr_s["prev_symbol"] = tr_s.groupby("tape_day")["symbol"].shift(1)
    hop = tr_s[tr_s["prev_seller"] == tr_s["buyer"]].copy()
    hop["chain"] = hop["prev_buyer"].astype(str) + "->" + hop["buyer"].astype(str) + "->" + hop["seller"].astype(str)
    hop["dt"] = hop["timestamp"].astype(int) - hop["prev_ts"].astype(int)
    hop = hop[hop["dt"] <= 5000]  # same burst or nearby timestamps (tape units)
    hop_counts = hop.groupby("chain").size().sort_values(ascending=False).reset_index(name="n")
    hop_counts.to_csv(OUT / "twohop_chain_counts.csv", index=False)

    hop_fwd: list[dict] = []
    for tape_day in DAYS:
        px = load_prices_day(tape_day)
        mids, prod_index, ts_to_idx = build_mid_matrix(px)
        ui = prod_index["VELVETFRUIT_EXTRACT"]
        for _, r in hop[hop["tape_day"] == tape_day].iterrows():
            ti = ts_to_idx.get(int(r["timestamp"]))
            if ti is None:
                continue
            for k in HORIZONS:
                hop_fwd.append(
                    {
                        "tape_day": tape_day,
                        "chain": r["chain"],
                        "k": k,
                        "fwd_extract": forward_delta(mids, ti, ui, k),
                    }
                )
    if hop_fwd:
        hf = pd.DataFrame(hop_fwd)
        hf_sum = hf.groupby(["chain", "k"]).agg(n=("fwd_extract", "count"), mean=("fwd_extract", "mean")).reset_index()
        hf_sum.sort_values(["k", "n"], ascending=[True, False]).to_csv(OUT / "twohop_chain_fwd_extract.csv", index=False)

    # --- Baseline: cell mean fwd_same for (buyer, seller, symbol, spread_regime) ---
    cell_rows: list[dict] = []
    for tape_day in DAYS:
        px = load_prices_day(tape_day)
        mids, prod_index, ts_to_idx = build_mid_matrix(px)
        sub = m[m["tape_day"] == tape_day]
        for _, r in sub.iterrows():
            sym = str(r["symbol"])
            if sym not in prod_index:
                continue
            ti = ts_to_idx.get(int(r["timestamp"]))
            if ti is None:
                continue
            sj = prod_index[sym]
            for k in HORIZONS:
                cell_rows.append(
                    {
                        "buyer": str(r["buyer"]),
                        "seller": str(r["seller"]),
                        "symbol": sym,
                        "spread_regime": r["spread_regime"],
                        "k": k,
                        "tape_day": tape_day,
                        "fwd_same": forward_delta(mids, ti, sj, k),
                    }
                )
    cell_df = pd.DataFrame(cell_rows)
    baseline = (
        cell_df.groupby(["buyer", "seller", "symbol", "spread_regime", "k"], dropna=False)["fwd_same"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "baseline_mean", "count": "cell_n"})
    )
    cell_df = cell_df.merge(
        baseline,
        on=["buyer", "seller", "symbol", "spread_regime", "k"],
        how="left",
    )
    cell_df["residual"] = cell_df["fwd_same"] - cell_df["baseline_mean"]
    cell_df.to_csv(OUT / "per_print_with_baseline.csv", index=False)
    resid_sum = (
        cell_df.groupby(["buyer", "seller", "symbol", "k", "tape_day"])["residual"]
        .agg(["mean", "count"])
        .reset_index()
    )
    resid_sum.to_csv(OUT / "pair_baseline_residuals.csv", index=False)

    # --- Graph buyer -> seller ---
    tr["notional"] = (tr["price"].astype(float) * tr["quantity"].astype(float)).abs()
    edges = (
        tr.groupby(["buyer", "seller"])
        .agg(count=("symbol", "size"), notional=("notional", "sum"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    edges.to_csv(OUT / "graph_edges.csv", index=False)
    top_pairs_txt = "\n".join(
        f"{row['buyer']} -> {row['seller']}: n={row['count']}, notional={row['notional']:.1f}"
        for _, row in edges.head(20).iterrows()
    )
    (OUT / "graph_top_pairs.txt").write_text(top_pairs_txt + "\n", encoding="utf-8")

    # Hub scores
    out_h = Counter()
    in_h = Counter()
    for _, r in edges.iterrows():
        out_h[str(r["buyer"])] += int(r["count"])
        in_h[str(r["seller"])] += int(r["count"])
    hubs = [{"name": n, "out": out_h[n], "in": in_h[n], "total": out_h[n] + in_h[n]} for n in set(out_h) | set(in_h)]
    hubs.sort(key=lambda x: -x["total"])
    (OUT / "graph_hubs.json").write_text(json.dumps(hubs, indent=2), encoding="utf-8")

    # --- Bursts ---
    burst_keys = m.loc[m["burst_n"] >= 4, ["tape_day", "timestamp"]].drop_duplicates()
    burst_keys = burst_keys.assign(is_burst=1)
    burst_detail = (
        m.merge(burst_keys, on=["tape_day", "timestamp"], how="inner")
        .groupby(["tape_day", "timestamp"])
        .agg(
            n_prints=("symbol", "size"),
            n_sym=("symbol", pd.Series.nunique),
            buyer_mode=("buyer", lambda s: s.mode().iloc[0] if len(s) else ""),
            seller_mode=("seller", lambda s: s.mode().iloc[0] if len(s) else ""),
        )
        .reset_index()
    )
    burst_detail.to_csv(OUT / "burst_events.csv", index=False)

    # Forward extract + core vouchers after burst timestamps vs random controls (same RNG + control count per day)
    rng = np.random.default_rng(0)
    burst_fwd: list[dict] = []
    burst_core: list[dict] = []
    core_syms = ("VEV_5200", "VEV_5300")
    for tape_day in DAYS:
        px = load_prices_day(tape_day)
        mids, prod_index, ts_to_idx = build_mid_matrix(px)
        ui = prod_index["VELVETFRUIT_EXTRACT"]
        ts_burst = burst_detail.loc[burst_detail["tape_day"] == tape_day, "timestamp"].unique()
        all_ts = list(ts_to_idx.keys())
        controls = rng.choice(all_ts, size=min(len(ts_burst), 200), replace=False) if len(all_ts) else []
        for ts in ts_burst:
            ti = ts_to_idx.get(int(ts))
            if ti is None:
                continue
            for k in HORIZONS:
                burst_fwd.append(
                    {
                        "tape_day": tape_day,
                        "kind": "burst",
                        "k": k,
                        "fwd_extract": forward_delta(mids, ti, ui, k),
                    }
                )
                for cs in core_syms:
                    cj = prod_index[cs]
                    burst_core.append(
                        {
                            "tape_day": tape_day,
                            "kind": "burst",
                            "k": k,
                            "symbol": cs,
                            "fwd_mid": forward_delta(mids, ti, cj, k),
                        }
                    )
        for ts in controls:
            ti = ts_to_idx.get(int(ts))
            if ti is None:
                continue
            for k in HORIZONS:
                burst_fwd.append(
                    {
                        "tape_day": tape_day,
                        "kind": "control",
                        "k": k,
                        "fwd_extract": forward_delta(mids, ti, ui, k),
                    }
                )
                for cs in core_syms:
                    cj = prod_index[cs]
                    burst_core.append(
                        {
                            "tape_day": tape_day,
                            "kind": "control",
                            "k": k,
                            "symbol": cs,
                            "fwd_mid": forward_delta(mids, ti, cj, k),
                        }
                    )
    bf = pd.DataFrame(burst_fwd)
    bf.to_csv(OUT / "burst_forward_extract.csv", index=False)
    bc = pd.DataFrame(burst_core)
    bc.to_csv(OUT / "burst_forward_vev5200_vev5300_rows.csv", index=False)
    if len(bc):
        bc_sum = (
            bc.groupby(["tape_day", "kind", "k", "symbol"], dropna=False)["fwd_mid"]
            .agg(mean="mean", n="count", med="median")
            .reset_index()
        )
        bc_sum.to_csv(OUT / "burst_forward_core_vev_summary.csv", index=False)

    # --- Passive adverse: when seller is passive (sell_aggr means seller hit bid — seller aggressed; passive seller is when buy_aggr -> buyer hits ask, seller passive)
    # Phase text: markout after prints where U is aggressor hurts passive side.
    adv_rows: list[dict] = []
    for tape_day in DAYS:
        px = load_prices_day(tape_day)
        mids, prod_index, ts_to_idx = build_mid_matrix(px)
        sub = m[m["tape_day"] == tape_day]
        for _, r in sub.iterrows():
            sym = str(r["symbol"])
            if sym not in prod_index:
                continue
            ti = ts_to_idx.get(int(r["timestamp"]))
            if ti is None:
                continue
            sj = prod_index[sym]
            ag = r["aggression"]
            passive = None
            if ag == "buy_aggr":
                passive = str(r["seller"])
            elif ag == "sell_aggr":
                passive = str(r["buyer"])
            if passive is None:
                continue
            for k in HORIZONS:
                adv_rows.append(
                    {
                        "tape_day": tape_day,
                        "passive_party": passive,
                        "aggressor_side": ag,
                        "pair": f"{r['buyer']}->{r['seller']}",
                        "symbol": sym,
                        "k": k,
                        "fwd_same": forward_delta(mids, ti, sj, k),
                    }
                )
    adv_df = pd.DataFrame(adv_rows)
    if len(adv_df):
        adv_g = (
            adv_df.groupby(["pair", "passive_party", "k"])
            .agg(n=("fwd_same", "count"), mean_fwd=("fwd_same", "mean"))
            .reset_index()
            .sort_values("mean_fwd")
        )
        adv_g.to_csv(OUT / "passive_adverse_by_pair.csv", index=False)

    # --- Summary JSON for analysis.json ---
    def top_candidates(pdf: pd.DataFrame, n: int = 8) -> list[dict]:
        x = pdf[(pdf["n"] >= 40) & (pdf["k"] == 20)].copy()
        if x.empty:
            x = pdf[(pdf["n"] >= 25) & (pdf["k"] == 20)].copy()
        x = x.sort_values("t_stat", ascending=False, na_position="last").head(n)
        return x.to_dict(orient="records")

    burst_core_k20 = []
    if len(bc):
        burst_core_k20 = (
            bc[bc["k"] == 20]
            .groupby(["tape_day", "kind", "symbol"], dropna=False)["fwd_mid"]
            .agg(["mean", "count"])
            .reset_index()
            .to_dict(orient="records")
        )

    summary = {
        "n_trade_rows": int(len(tr)),
        "n_participant_cells_pooled": int(len(pooled)),
        "burst_event_count": int(len(burst_detail)),
        "top_positive_markout_candidates_k20": top_candidates(pooled),
        "burst_vs_control_extract_k20": bf[bf["k"] == 20]
        .groupby(["tape_day", "kind"])["fwd_extract"]
        .agg(["mean", "count"])
        .reset_index()
        .to_dict(orient="records"),
        "burst_vs_control_core_vev_k20": burst_core_k20,
        "edges_top3": edges.head(3).to_dict(orient="records"),
    }
    (OUT / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote outputs to", OUT)


if __name__ == "__main__":
    main()
