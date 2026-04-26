#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned markouts, baseline cells, graph,
bursts, passive proxy (see round4work/suggested direction.txt).

Reads Prosperity4Data/ROUND_4/prices_round_4_day_*.csv and trades_round_4_day_*.csv.
Horizon K = number of **tape timestamp steps** (same index grid as STRATEGY K=20 style).

Writes under manual_traders/R4/r3v_gamma_scalping_extract_20/analysis_outputs/:
  r4_phase1_participant_summary.csv
  r4_phase1_pair_symbol_horizon.csv
  r4_phase1_graph_edges.csv
  r4_phase1_two_hop_motifs.csv
  r4_phase1_burst_event_study.csv
  r4_phase1_baseline_residual_summary.csv
  r4_phase1_adverse_summary.csv
  r4_phase1_run_log.txt
"""
from __future__ import annotations

import bisect
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
KS = (5, 20, 100)
PRODUCTS_FOCUS = [
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    "VEV_5200",
    "VEV_5300",
    "VEV_5000",
    "VEV_5400",
]


def _load_prices() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        if "day" not in df.columns:
            df["day"] = d
        frames.append(df)
    if not frames:
        raise SystemExit(f"No price files in {DATA}")
    return pd.concat(frames, ignore_index=True)


def _load_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    if not frames:
        raise SystemExit(f"No trade files in {DATA}")
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    return tr


def _prep_price_panel(px: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[tuple[int, int, str], float], dict[tuple[int, int, str], float]]:
    """Per day: sorted unique timestamps. Lookup mid and spread by (day, ts, product)."""
    px["mid_price"] = pd.to_numeric(px["mid_price"], errors="coerce")
    for c in ("bid_price_1", "ask_price_1"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px["spread"] = (px["ask_price_1"] - px["bid_price_1"]).astype(float)

    ts_sorted: dict[int, np.ndarray] = {}
    mid_lk: dict[tuple[int, int, str], float] = {}
    spr_lk: dict[tuple[int, int, str], float] = {}
    for d in DAYS:
        sub = px[px["day"] == d]
        tsu = np.sort(sub["timestamp"].unique())
        ts_sorted[d] = tsu
        g = sub.groupby(["timestamp", "product"]).first(numeric_only=False)
        for (ts, prod), row in g.iterrows():
            mid_lk[(d, int(ts), str(prod))] = float(row["mid_price"])
            spr_lk[(d, int(ts), str(prod))] = float(row["spread"])
    return ts_sorted, mid_lk, spr_lk


def _fwd_ts(ts_sorted: np.ndarray, t: int, k: int) -> int | None:
    i = bisect.bisect_left(ts_sorted, t)
    if i >= len(ts_sorted) or ts_sorted[i] != t:
        return None
    j = i + k
    if j >= len(ts_sorted):
        return None
    return int(ts_sorted[j])


def _session_quartile(day: int, ts: int, ts_sorted: np.ndarray) -> int:
    """0..3 quartile of timestamp rank within day (coarse session bucket)."""
    i = bisect.bisect_left(ts_sorted, ts)
    if len(ts_sorted) <= 1:
        return 0
    q = int(4 * i / max(1, len(ts_sorted) - 1))
    return min(3, max(0, q))


def _spread_bucket(d: int, ts: int, sym: str, spr_lk: dict, px_day_spread_quantile: dict) -> str:
    sp = spr_lk.get((d, ts, sym))
    if sp is None or math.isnan(sp):
        return "unk"
    lo, hi = px_day_spread_quantile[(d, sym)]
    if sp <= lo:
        return "tight"
    if sp >= hi:
        return "wide"
    return "mid"


def main() -> int:
    log_path = OUT / "r4_phase1_run_log.txt"
    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        print(msg)

    px = _load_prices()
    tr = _load_trades()
    ts_sorted, mid_lk, spr_lk = _prep_price_panel(px)

    # Rebuild bid/ask lookup for aggressor inference
    bb_lk: dict[tuple[int, int, str], float] = {}
    ba_lk: dict[tuple[int, int, str], float] = {}
    for _, row in px.iterrows():
        key = (int(row["day"]), int(row["timestamp"]), str(row["product"]))
        bb_lk[key] = float(row["bid_price_1"]) if pd.notna(row["bid_price_1"]) else float("nan")
        ba_lk[key] = float(row["ask_price_1"]) if pd.notna(row["ask_price_1"]) else float("nan")

    # Per (day, symbol) spread quantiles for tight/wide
    px_day_spread_quantile: dict[tuple[int, str], tuple[float, float]] = {}
    for d in DAYS:
        for sym in px["product"].unique():
            s = px[(px["day"] == d) & (px["product"] == sym)]["spread"].dropna()
            if len(s) < 20:
                continue
            lo, hi = float(s.quantile(0.33)), float(s.quantile(0.66))
            px_day_spread_quantile[(d, str(sym))] = (lo, hi)

    def aggressor(d: int, ts: int, sym: str, price: float) -> str:
        bb, ba = bb_lk.get((d, ts, sym)), ba_lk.get((d, ts, sym))
        if bb is None or ba is None or math.isnan(bb) or math.isnan(ba):
            return "unk"
        if price >= ba:
            return "buy_agg"
        if price <= bb:
            return "sell_agg"
        return "mid_passive"

    rows: list[dict] = []
    skipped = 0
    for _, r in tr.iterrows():
        d, ts, sym = int(r["day"]), int(r["timestamp"]), str(r["symbol"])
        price = float(r["price"])
        buyer, seller = str(r["buyer"]), str(r["seller"])
        qty = int(r["quantity"])
        tsu = ts_sorted.get(d)
        if tsu is None:
            skipped += 1
            continue
        m0 = mid_lk.get((d, ts, sym))
        if m0 is None or math.isnan(m0):
            skipped += 1
            continue
        ag = aggressor(d, ts, sym, price)
        sq = _spread_bucket(d, ts, sym, spr_lk, px_day_spread_quantile)
        sess = _session_quartile(d, ts, tsu)

        ext0 = mid_lk.get((d, ts, "VELVETFRUIT_EXTRACT"))
        hyd0 = mid_lk.get((d, ts, "HYDROGEL_PACK"))

        for K in KS:
            fts = _fwd_ts(tsu, ts, K)
            if fts is None:
                continue
            m1 = mid_lk.get((d, fts, sym))
            d_same = (m1 - m0) if m1 is not None and not math.isnan(m1) else float("nan")
            ext1 = mid_lk.get((d, fts, "VELVETFRUIT_EXTRACT"))
            hyd1 = mid_lk.get((d, fts, "HYDROGEL_PACK"))
            d_ext = (ext1 - ext0) if (
                ext0 is not None and ext1 is not None and not math.isnan(ext0) and not math.isnan(ext1)
            ) else float("nan")
            d_hyd = (hyd1 - hyd0) if (
                hyd0 is not None and hyd1 is not None and not math.isnan(hyd0) and not math.isnan(hyd1)
            ) else float("nan")
            rows.append(
                {
                    "day": d,
                    "timestamp": ts,
                    "symbol": sym,
                    "buyer": buyer,
                    "seller": seller,
                    "qty": qty,
                    "aggressor": ag,
                    "spread_bucket": sq,
                    "session_q": sess,
                    "K": K,
                    "d_mid_same": d_same,
                    "d_mid_extract": d_ext,
                    "d_mid_hydro": d_hyd,
                }
            )

    df = pd.DataFrame(rows)
    log(f"Trade rows: {len(tr):,} | markout rows (× horizons): {len(df):,} | skipped (no mid): {skipped:,}")

    # --- 1) Participant-level summary (U involved as buyer or seller) ---
    def tstat(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 8:
            return float("nan")
        m, s = float(np.mean(x)), float(np.std(x, ddof=1))
        if s < 1e-12:
            return float("nan")
        return m / (s / math.sqrt(len(x)))

    part_rows: list[dict] = []
    for U in sorted(set(tr["buyer"].astype(str)) | set(tr["seller"].astype(str))):
        for role in ("buyer", "seller"):
            sub = df[df["buyer"] == U] if role == "buyer" else df[df["seller"] == U]
            if sub.empty:
                continue
            for (sym, K, ag, sb), g in sub.groupby(["symbol", "K", "aggressor", "spread_bucket"]):
                x = g["d_mid_same"].values
                if len(x) < 15:
                    continue
                part_rows.append(
                    {
                        "U": U,
                        "role": role,
                        "symbol": sym,
                        "K": K,
                        "aggressor": ag,
                        "spread_bucket": sb,
                        "n": len(x),
                        "mean_d_mid": float(np.nanmean(x)),
                        "median_d_mid": float(np.nanmedian(x)),
                        "pos_frac": float(np.mean(x > 0)),
                        "t_stat": tstat(x.astype(float)),
                    }
                )
    part_df = pd.DataFrame(part_rows)
    part_path = OUT / "r4_phase1_participant_summary.csv"
    part_df.sort_values(["n", "t_stat"], ascending=[False, False]).to_csv(part_path, index=False)
    log(f"Wrote {part_path} ({len(part_df):,} aggregates)")

    # Stability across days: Mark 01 buyer, VEV_5300, K=20, buy_agg, tight
    stab = (
        df[(df["buyer"] == "Mark 01") & (df["symbol"] == "VEV_5300") & (df["K"] == 20) & (df["aggressor"] == "buy_agg")]
        .groupby("day")["d_mid_same"]
        .agg(["count", "mean"])
        .reset_index()
    )
    stab.to_csv(OUT / "r4_phase1_example_mark01_vev5300_k20_buyagg_by_day.csv", index=False)

    # --- 2) Baseline: cell means (buyer, seller, symbol, spread_bucket, K) ---
    cell = (
        df.groupby(["buyer", "seller", "symbol", "spread_bucket", "K"])["d_mid_same"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    cell = cell.rename(columns={"count": "n", "mean": "cell_mean", "std": "cell_std"})
    cell_path = OUT / "r4_phase1_baseline_cells.csv"
    cell.to_csv(cell_path, index=False)
    log(f"Wrote {cell_path}")

    # Residuals: merge each row with cell mean
    mrg = df.merge(
        cell,
        on=["buyer", "seller", "symbol", "spread_bucket", "K"],
        how="left",
        suffixes=("", "_cell"),
    )
    mrg["residual"] = mrg["d_mid_same"] - mrg["cell_mean"]
    resid_sum = (
        mrg.groupby(["buyer", "seller", "symbol", "K"])
        .agg(n=("residual", "count"), mean_resid=("residual", "mean"), std_resid=("residual", "std"))
        .reset_index()
    )
    resid_sum = resid_sum[resid_sum["n"] >= 30].sort_values("mean_resid", key=np.abs, ascending=False)
    resid_path = OUT / "r4_phase1_baseline_residual_summary.csv"
    resid_sum.head(200).to_csv(resid_path, index=False)
    log(f"Wrote {resid_path}")

    # --- 3) Graph buyer → seller ---
    pair_counts = tr.groupby(["buyer", "seller"]).size().reset_index(name="n")
    pair_notional = tr.assign(notional=tr["price"] * tr["quantity"]).groupby(["buyer", "seller"])["notional"].sum().reset_index()
    edges = pair_counts.merge(pair_notional, on=["buyer", "seller"], how="left")
    edges = edges.sort_values("n", ascending=False)
    edges_path = OUT / "r4_phase1_graph_edges.csv"
    edges.to_csv(edges_path, index=False)
    log(f"Wrote {edges_path}")

    # 2-hop motifs A→B→C by counts on consecutive trades same day (coarse): use trade order
    tr_sorted = tr.sort_values(["day", "timestamp", "symbol"]).reset_index(drop=True)
    hops: Counter[tuple[str, str, str]] = Counter()
    for d in DAYS:
        sub = tr_sorted[tr_sorted["day"] == d]
        buyers = sub["buyer"].astype(str).tolist()
        sellers = sub["seller"].astype(str).tolist()
        for i in range(len(sub) - 1):
            u1, v1 = buyers[i], sellers[i]
            u2, v2 = buyers[i + 1], sellers[i + 1]
            # Two-hop: u1 -> v1 == u2 -> v2 (seller of print i is buyer of print i+1)
            if v1 == u2:
                hops[(u1, v1, v2)] += 1
    hop_rows = [{"buyer": a, "mid": b, "seller": c, "count": n} for (a, b, c), n in hops.most_common(50)]
    pd.DataFrame(hop_rows).to_csv(OUT / "r4_phase1_two_hop_motifs.csv", index=False)
    log("Wrote r4_phase1_two_hop_motifs.csv")

    # --- 4) Bursts: same (day, timestamp) with >=2 trades ---
    burst_key = tr.groupby(["day", "timestamp"]).agg(
        n=("symbol", "count"),
        n_sym=("symbol", lambda x: x.nunique()),
        buyer_mode=("buyer", lambda x: x.mode().iloc[0] if len(x) else ""),
        seller_mode=("seller", lambda x: x.mode().iloc[0] if len(x) else ""),
    ).reset_index()
    burst_key = burst_key[burst_key["n"] >= 3].copy()
    burst_times = set(zip(burst_key["day"], burst_key["timestamp"]))
    df["burst"] = df.apply(lambda r: (r["day"], r["timestamp"]) in burst_times, axis=1)

    def burst_event_study() -> pd.DataFrame:
        out = []
        for K in KS:
            for burst_flag, name in [(True, "burst"), (False, "control")]:
                sub = df[(df["K"] == K) & (df["burst"] == burst_flag)]
                if sub.empty:
                    continue
                out.append(
                    {
                        "K": K,
                        "cohort": name,
                        "n": len(sub),
                        "mean_d_extract": float(sub["d_mid_extract"].mean()),
                        "mean_d_vev5200": float(sub.loc[sub["symbol"] == "VEV_5200", "d_mid_same"].mean()),
                    }
                )
        return pd.DataFrame(out)

    burst_df = burst_event_study()
    burst_df.to_csv(OUT / "r4_phase1_burst_event_study.csv", index=False)
    log(f"Wrote r4_phase1_burst_event_study.csv\n{burst_df}")

    # --- 5) Adverse selection proxy: markout same symbol after print, split aggressor ---
    adv = (
        df.groupby(["seller", "buyer", "aggressor", "K"])["d_mid_same"]
        .agg(n="count", mean_d="mean")
        .reset_index()
    )
    adv = adv[adv["n"] >= 40].sort_values("mean_d")
    adv.to_csv(OUT / "r4_phase1_adverse_summary.csv", index=False)
    log(f"Wrote r4_phase1_adverse_summary.csv ({len(adv):,} rows)")

    # Pair × symbol × K quick view
    def _tgrp(s: pd.Series) -> float:
        return tstat(s.astype(float).values)

    psh = (
        df.groupby(["buyer", "seller", "symbol", "K"])["d_mid_same"]
        .agg(n="count", mean="mean", t=_tgrp)
        .reset_index()
    )
    psh = psh[psh["n"] >= 25].sort_values("mean", ascending=False)
    psh.to_csv(OUT / "r4_phase1_pair_symbol_horizon.csv", index=False)
    log(f"Wrote r4_phase1_pair_symbol_horizon.csv")

    # Day-stability slices (Phase 1 gate: multi-day)
    key67 = df[
        (df["buyer"] == "Mark 67")
        & (df["symbol"] == "VELVETFRUIT_EXTRACT")
        & (df["K"] == 5)
        & (df["aggressor"] == "buy_agg")
        & (df["spread_bucket"] == "tight")
    ]
    key67.groupby("day")["d_mid_same"].agg(["count", "mean", "median"]).reset_index().to_csv(
        OUT / "r4_phase1_mark67_extract_buyagg_k5_by_day.csv", index=False
    )
    log("Wrote r4_phase1_mark67_extract_buyagg_k5_by_day.csv")

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
