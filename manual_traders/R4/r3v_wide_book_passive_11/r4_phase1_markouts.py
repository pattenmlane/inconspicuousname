#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mid markouts (tape only).

Horizon K: **K rows ahead** in the prices tape for the same (day, product), sorted by
timestamp (consecutive market snapshots). Cross-asset: VELVETFRUIT_EXTRACT forward from
the same row index.

Outputs under: manual_traders/R4/r3v_wide_book_passive_11/analysis_outputs_r4_phase1/

Run from repo root:
  python3 manual_traders/R4/r3v_wide_book_passive_11/r4_phase1_markouts.py
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# .../manual_traders/R4/<id>/this_script.py -> repo root is parents[3]
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs_r4_phase1"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
KS = [5, 20, 100]
EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
N_BOOT_PARTICIPANT = 400
N_BOOT_STRAT = 250


def _trade_paths() -> list[tuple[int, Path]]:
    paths: list[tuple[int, Path]] = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            paths.append((d, p))
    return paths


def load_prices() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    if not frames:
        raise SystemExit(f"No price files under {DATA}")
    pr = pd.concat(frames, ignore_index=True)
    pr["mid"] = pd.to_numeric(pr["mid_price"], errors="coerce")
    bb = pd.to_numeric(pr["bid_price_1"], errors="coerce")
    aa = pd.to_numeric(pr["ask_price_1"], errors="coerce")
    pr["spread"] = aa - bb
    pr["bid1"] = bb
    pr["ask1"] = aa
    return pr


def add_forward_mids(pr: pd.DataFrame) -> pd.DataFrame:
    pr = pr.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)
    for K in KS:
        col = f"fwd_mid_{K}"
        pr[col] = (
            pr.groupby(["day", "product"], group_keys=False)["mid"]
            .transform(lambda s: s.shift(-K) - s)
        )
    return pr


def load_trades() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d, p in _trade_paths():
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    if not frames:
        raise SystemExit("No trade files")
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    tr["symbol"] = tr["symbol"].astype(str)
    tr["buyer"] = tr["buyer"].astype(str)
    tr["seller"] = tr["seller"].astype(str)
    return tr


def merge_trades_prices(tr: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    px = pr.rename(columns={"product": "symbol"})
    sym_cols = [
        "day",
        "timestamp",
        "symbol",
        "mid",
        "spread",
        "bid1",
        "ask1",
    ] + [f"fwd_mid_{K}" for K in KS]
    m = tr.merge(px[sym_cols], on=["day", "timestamp", "symbol"], how="left")
    ex = px.loc[px["symbol"] == EXTRACT, ["day", "timestamp"] + [f"fwd_mid_{K}" for K in KS]]
    ren = {f"fwd_mid_{K}": f"extract_fwd_mid_{K}" for K in KS}
    ex = ex.rename(columns=ren)
    m = m.merge(ex, on=["day", "timestamp"], how="left")
    hy = px.loc[px["symbol"] == HYDRO, ["day", "timestamp"] + [f"fwd_mid_{K}" for K in KS]]
    hy = hy.rename(columns={f"fwd_mid_{K}": f"hydro_fwd_mid_{K}" for K in KS})
    m = m.merge(hy, on=["day", "timestamp"], how="left")
    return m


def aggressor_side(row: pd.Series) -> str:
    """Rough touch aggression: buyer aggressive if price >= ask1; seller if price <= bid1."""
    bid1, ask1, price = row["bid1"], row["ask1"], row["price"]
    if pd.isna(bid1) or pd.isna(ask1) or pd.isna(price):
        return "unknown"
    if price >= ask1:
        return "buy_aggr"
    if price <= bid1:
        return "sell_aggr"
    return "mid_passive"


def burst_stats(tr: pd.DataFrame) -> pd.DataFrame:
    g = tr.groupby(["day", "timestamp"]).agg(
        n_prints=("symbol", "size"),
        symbols=("symbol", lambda s: ",".join(sorted(set(s)))),
        buyer_mode=("buyer", lambda s: s.mode().iloc[0] if len(s) else ""),
        seller_mode=("seller", lambda s: s.mode().iloc[0] if len(s) else ""),
    )
    g = g.reset_index()
    g["burst"] = g["n_prints"] > 1
    return g


def one_sample_t_mean0(a: np.ndarray) -> float:
    """t-statistic for H0: E[x]=0 (two-sided inference uses normal approx for large n)."""
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float("nan")
    m = float(a.mean())
    s = float(a.std(ddof=1))
    if s == 0:
        return float("nan")
    return m / (s / math.sqrt(len(a)))


def bootstrap_mean_ci(a: np.ndarray, n_boot: int = 400, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI for mean; returns (nan,nan) if n < 10."""
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(a[idx].mean())
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def t_stat_welch(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan")
    return (ma - mb) / se


def main() -> None:
    pr = load_prices()
    pr = add_forward_mids(pr)
    tr = load_trades()
    m = merge_trades_prices(tr, pr)
    m["aggressor"] = m.apply(aggressor_side, axis=1)

    # Spread quantile per symbol (pooled days)
    m["spread_q"] = m.groupby("symbol")["spread"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["tight", "mid", "wide"], duplicates="drop")
    )

    # Session tertile by timestamp within day
    m["ts_tert"] = m.groupby("day")["timestamp"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["early", "mid", "late"], duplicates="drop")
    )

    burst = burst_stats(tr)
    m = m.merge(
        burst[["day", "timestamp", "n_prints"]].rename(columns={"n_prints": "burst_n"}),
        on=["day", "timestamp"],
        how="left",
    )
    m["burst"] = (m["burst_n"] > 1).fillna(False)

    lines: list[str] = []

    # --- 1) Participant tables (per Mark, per aggressor subset) ---
    rows = []
    marks = sorted(set(m["buyer"]) | set(m["seller"]))
    for U in marks:
        for role in ("any", "buy_aggr", "sell_aggr", "mid_passive"):
            if role == "any":
                sub = m[(m["buyer"] == U) | (m["seller"] == U)]
            elif role == "buy_aggr":
                sub = m[(m["buyer"] == U) & (m["aggressor"] == "buy_aggr")]
            elif role == "sell_aggr":
                sub = m[(m["seller"] == U) & (m["aggressor"] == "sell_aggr")]
            else:
                sub = m[((m["buyer"] == U) | (m["seller"] == U)) & (m["aggressor"] == "mid_passive")]
            for K in KS:
                col = f"fwd_mid_{K}"
                ex_col = f"extract_fwd_mid_{K}"
                hy_col = f"hydro_fwd_mid_{K}"
                if col not in sub.columns:
                    continue
                x = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
                ex = pd.to_numeric(sub[ex_col], errors="coerce").dropna().to_numpy(dtype=float)
                if hy_col in sub.columns:
                    hy = pd.to_numeric(sub[hy_col], errors="coerce").dropna().to_numpy(dtype=float)
                else:
                    hy = np.array([], dtype=float)
                x_lo, x_hi = bootstrap_mean_ci(x, n_boot=N_BOOT_PARTICIPANT)
                ex_lo, ex_hi = bootstrap_mean_ci(ex, n_boot=N_BOOT_PARTICIPANT)
                hy_lo, hy_hi = bootstrap_mean_ci(hy, n_boot=N_BOOT_PARTICIPANT)
                rows.append(
                    {
                        "mark": U,
                        "role": role,
                        "K": K,
                        "n": len(sub),
                        "n_same_sym_fwd": int(x.shape[0]),
                        "mean_same_sym": float(x.mean()) if len(x) else float("nan"),
                        "median_same_sym": float(np.median(x)) if len(x) else float("nan"),
                        "t_same_sym_mean0": one_sample_t_mean0(x),
                        "ci_same_sym_lo": x_lo,
                        "ci_same_sym_hi": x_hi,
                        "frac_pos_same": float((x > 0).mean()) if len(x) else float("nan"),
                        "mean_extract_fwd": float(ex.mean()) if len(ex) else float("nan"),
                        "median_extract_fwd": float(np.median(ex)) if len(ex) else float("nan"),
                        "t_extract_mean0": one_sample_t_mean0(ex),
                        "ci_extract_lo": ex_lo,
                        "ci_extract_hi": ex_hi,
                        "frac_pos_extract": float((ex > 0).mean()) if len(ex) else float("nan"),
                        "mean_hydro_fwd": float(hy.mean()) if len(hy) else float("nan"),
                        "median_hydro_fwd": float(np.median(hy)) if len(hy) else float("nan"),
                        "t_hydro_mean0": one_sample_t_mean0(hy),
                        "ci_hydro_lo": hy_lo,
                        "ci_hydro_hi": hy_hi,
                        "frac_pos_hydro": float((hy > 0).mean()) if len(hy) else float("nan"),
                    }
                )
    t1 = pd.DataFrame(rows)
    t1.to_csv(OUT / "01_participant_fwd_summary.csv", index=False)
    lines.append(f"Wrote {OUT / '01_participant_fwd_summary.csv'}")

    # Per-day same-symbol stability (any touch): coarse multi-day check for spec "stability across days"
    stab_rows: list[dict] = []
    for U in marks:
        sub_any = m[(m["buyer"] == U) | (m["seller"] == U)]
        for K in KS:
            col = f"fwd_mid_{K}"
            for day, sd in sub_any.groupby("day"):
                x = pd.to_numeric(sd[col], errors="coerce").dropna()
                if len(x) < 5:
                    continue
                stab_rows.append(
                    {
                        "mark": U,
                        "K": int(K),
                        "day": int(day),
                        "n": int(len(x)),
                        "mean_same_sym": float(x.mean()),
                        "median_same_sym": float(x.median()),
                        "frac_pos": float((x > 0).mean()),
                    }
                )
    if stab_rows:
        pd.DataFrame(stab_rows).to_csv(OUT / "09_participant_stability_by_day.csv", index=False)
        lines.append(f"Wrote {OUT / '09_participant_stability_by_day.csv'}")

    # Stratified: participant U touched trade (buyer or seller), symbol, spread_q, burst — K=20
    K = 20
    col = f"fwd_mid_{K}"
    strat_rows = []
    for U in marks:
        subu = m[(m["buyer"] == U) | (m["seller"] == U)]
        for (sym, sq, br), sub in subu.groupby(["symbol", "spread_q", "burst"]):
            x = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(x) < 25:
                continue
            xa = x.to_numpy(dtype=float)
            slo, shi = bootstrap_mean_ci(xa, n_boot=N_BOOT_STRAT, seed=hash((U, sym, sq, br)) % (2**31))
            strat_rows.append(
                {
                    "mark": U,
                    "symbol": sym,
                    "spread_q": str(sq),
                    "burst": bool(br),
                    "n": len(x),
                    "mean_fwd20": float(xa.mean()),
                    "median_fwd20": float(np.median(xa)),
                    "t_fwd20_mean0": one_sample_t_mean0(xa),
                    "ci_fwd20_lo": slo,
                    "ci_fwd20_hi": shi,
                    "frac_pos": float((xa > 0).mean()),
                }
            )
    if strat_rows:
        pd.DataFrame(strat_rows).sort_values("mean_fwd20", ascending=False).head(80).to_csv(
            OUT / "02_stratified_mark_symbol_fwd20_top80.csv", index=False
        )

    # Stratified: add **session tertile** (ts_tert) — K=20, same min_n
    sess_rows = []
    for U in marks:
        subu = m[(m["buyer"] == U) | (m["seller"] == U)]
        for (sym, sq, br, tert), sub in subu.groupby(["symbol", "spread_q", "burst", "ts_tert"]):
            x = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(x) < 25:
                continue
            xa = x.to_numpy(dtype=float)
            slo, shi = bootstrap_mean_ci(xa, n_boot=N_BOOT_STRAT, seed=hash((U, sym, sq, br, str(tert))) % (2**31))
            sess_rows.append(
                {
                    "mark": U,
                    "symbol": sym,
                    "spread_q": str(sq),
                    "burst": bool(br),
                    "session_tert": str(tert),
                    "n": len(x),
                    "mean_fwd20": float(xa.mean()),
                    "median_fwd20": float(np.median(xa)),
                    "t_fwd20_mean0": one_sample_t_mean0(xa),
                    "ci_fwd20_lo": slo,
                    "ci_fwd20_hi": shi,
                    "frac_pos": float((xa > 0).mean()),
                }
            )
    if sess_rows:
        pd.DataFrame(sess_rows).sort_values("mean_fwd20", ascending=False).head(80).to_csv(
            OUT / "12_stratified_mark_session_fwd20_top80.csv", index=False
        )
        lines.append(f"Wrote {OUT / '12_stratified_mark_session_fwd20_top80.csv'}")

    # Burst orchestrator: same (day,timestamp), >1 row — modal buyer/seller + symbol list
    burst_tbl = burst_stats(tr)
    burst_tbl = burst_tbl[burst_tbl["burst"]].copy()
    burst_tbl = burst_tbl.sort_values(["day", "timestamp"])
    burst_tbl.to_csv(OUT / "11_burst_orchestrator_timestamps.csv", index=False)
    lines.append(f"Wrote {OUT / '11_burst_orchestrator_timestamps.csv'}")

    # --- 2) Baseline cell means (buyer, seller, symbol) ---
    cell = (
        m.groupby(["buyer", "seller", "symbol"])[col]
        .agg(["count", "mean"])
        .reset_index()
    )
    cell = cell[cell["count"] >= 25].sort_values("mean", ascending=False)
    cell.to_csv(OUT / "03_pair_symbol_mean_fwd20.csv", index=False)
    global_mean = float(pd.to_numeric(m[col], errors="coerce").mean())
    cell["residual"] = cell["mean"] - global_mean
    cell.sort_values("residual", ascending=False).head(40).to_csv(
        OUT / "04_pair_symbol_residual_fwd20_top40.csv", index=False
    )

    # --- 3) Graph edges ---
    edge = m.groupby(["buyer", "seller"]).agg(n=("symbol", "size"), notional=("quantity", "sum"))
    edge = edge.reset_index().sort_values("n", ascending=False)
    edge.to_csv(OUT / "05_directed_edges_buyer_seller.csv", index=False)

    # Reciprocity: reverse edge count for top directed pairs (Phase 1 graph bullet)
    rec_rows = []
    edge_idx = edge.set_index(["buyer", "seller"])
    for _, r in edge.head(200).iterrows():
        a, b, n_ab = r["buyer"], r["seller"], int(r["n"])
        key = (b, a)
        n_ba = int(edge_idx.loc[key, "n"]) if key in edge_idx.index else 0
        rec_rows.append(
            {
                "buyer": a,
                "seller": b,
                "n_ab": n_ab,
                "n_ba_reverse": n_ba,
                "reciprocity_ratio": (n_ba / n_ab) if n_ab > 0 else float("nan"),
            }
        )
    pd.DataFrame(rec_rows).to_csv(OUT / "13_edge_reciprocity_top200_from_top200.csv", index=False)
    lines.append(f"Wrote {OUT / '13_edge_reciprocity_top200_from_top200.csv'}")

    # Hub scores: incident trade prints (out as buyer + in as seller) on directed edges
    out_buy = edge.groupby("buyer")["n"].sum()
    in_sell = edge.groupby("seller")["n"].sum()
    participants = sorted(set(edge["buyer"]) | set(edge["seller"]))
    hub = pd.DataFrame({"participant": participants})
    hub["out_as_buyer"] = hub["participant"].map(out_buy).fillna(0).astype(int)
    hub["in_as_seller"] = hub["participant"].map(in_sell).fillna(0).astype(int)
    hub["total_incident_prints"] = hub["out_as_buyer"] + hub["in_as_seller"]
    hub = hub.sort_values("total_incident_prints", ascending=False)
    hub.to_csv(OUT / "14_mark_hub_incident_edge_counts.csv", index=False)
    lines.append(f"Wrote {OUT / '14_mark_hub_incident_edge_counts.csv'}")

    two_hop = Counter()
    for _, r in edge.head(200).iterrows():
        a, b = r["buyer"], r["seller"]
        cont = edge[edge["buyer"] == b]
        for _, r2 in cont.head(30).iterrows():
            c = r2["seller"]
            two_hop[(a, b, c)] += int(min(r["n"], r2["n"]))
    top_tri = sorted(two_hop.items(), key=lambda x: -x[1])[:30]
    with open(OUT / "06_two_hop_motifs.txt", "w") as f:
        for (a, b, c), w in top_tri:
            f.write(f"{a} -> {b} -> {c}\tweight={w}\n")

    # --- 4) Bursts vs controls (extract forward at trade timestamps) ---
    br_sub = m[m["burst"]].copy()
    ctrl = m[~m["burst"]]
    ctrl = ctrl.sample(min(max(1, len(br_sub)), len(ctrl)), random_state=1)
    burst_lines = ["Burst vs control (VELVETFRUIT_EXTRACT forward, same row index K)\n"]
    for K in KS:
        ex_col = f"extract_fwd_mid_{K}"
        a = pd.to_numeric(br_sub[ex_col], errors="coerce").dropna().values
        b = pd.to_numeric(ctrl[ex_col], errors="coerce").dropna().values
        burst_lines.append(
            f"K={K}  burst n={len(a)} mean={a.mean():.5g}  control n={len(b)} mean={b.mean():.5g}  "
            f"welch_t={t_stat_welch(a, b):.3f}\n"
        )
    Path(OUT / "07_burst_event_study_extract.txt").write_text("".join(burst_lines))

    # --- 5) Adverse selection proxy: aggressor vs forward (same symbol) ---
    adv = []
    for ag, sub in m.groupby("aggressor"):
        if ag == "unknown":
            continue
        x = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) < 50:
            continue
        lo, hi = bootstrap_mean_ci(x, n_boot=N_BOOT_STRAT, seed=hash(ag) % (2**31))
        adv.append(
            {
                "aggressor": ag,
                "n": len(x),
                "mean_fwd20": float(x.mean()),
                "median_fwd20": float(np.median(x)),
                "t_fwd20_mean0": one_sample_t_mean0(x),
                "ci_fwd20_lo": lo,
                "ci_fwd20_hi": hi,
                "frac_pos": float((x > 0).mean()),
            }
        )
    pd.DataFrame(adv).to_csv(OUT / "08_aggressor_fwd20_same_symbol.csv", index=False)

    # Human-readable summary
    summary = []
    summary.append("Round 4 Phase 1 — automated summary (see CSVs in same folder)\n")
    summary.append(f"Global mean same-symbol fwd_mid_20: {global_mean:.6g}\n\n")
    summary.append("Top directed pairs by count:\n")
    for _, r in edge.head(12).iterrows():
        summary.append(f"  {r['buyer']} -> {r['seller']}: n={int(r['n'])}\n")
    summary.append("\nAggressor bucket same-symbol fwd20:\n")
    summary.append(Path(OUT / "08_aggressor_fwd20_same_symbol.csv").read_text())
    summary.append("\nBurst event study:\n")
    summary.append(Path(OUT / "07_burst_event_study_extract.txt").read_text())
    Path(OUT / "00_README_PHASE1.txt").write_text("".join(summary))
    print("Done. Outputs in", OUT)


if __name__ == "__main__":
    main()
