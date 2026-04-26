#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned markouts (suggested direction.txt).

Horizon definition: **K ticks** = move **K steps** forward on that product's **distinct**
price timestamps within the same tape day (sorted ascending). Forward mid change for a trade
at time T uses mid at the last price row with timestamp <= T, then mid at the row
**K steps** ahead in that symbol-day series (same convention for cross-assets at aligned
timestamps).

Aggressive side: at trade timestamp, merge bid1/ask1 from prices; **aggressive buy** if
trade price >= ask1; **aggressive sell** if price <= bid1; else **inside_spread**.

Outputs under manual_traders/R4/r3v_synthetic_local_vol_surface_15/outputs_r4_phase1/
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs_r4_phase1"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
PRODUCTS_FOCUS = [
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
K_HORIZONS = (5, 20, 100)


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = int(d)
        frames.append(df)
    if not frames:
        raise SystemExit("No price files")
    pr = pd.concat(frames, ignore_index=True)
    pr["mid"] = pd.to_numeric(pr["mid_price"], errors="coerce")
    pr["bid1"] = pd.to_numeric(pr["bid_price_1"], errors="coerce")
    pr["ask1"] = pd.to_numeric(pr["ask_price_1"], errors="coerce")
    pr["spread"] = pr["ask1"] - pr["bid1"]
    return pr


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = int(d)
        frames.append(df)
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    return tr


def build_ts_mid_index(pr: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(day, sym) -> (ts_sorted_unique, mid_at_ts, bid1, ask1) last row per ts if dupes."""
    out: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for (d, sym), g in pr.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        # last observation per timestamp
        g2 = g.groupby("timestamp", as_index=False).last()
        ts = g2["timestamp"].to_numpy(dtype=np.int64)
        mid = g2["mid"].to_numpy(dtype=np.float64)
        bid = g2["bid1"].to_numpy(dtype=np.float64)
        ask = g2["ask1"].to_numpy(dtype=np.float64)
        out[(int(d), str(sym))] = (ts, mid, bid, ask)
    return out


def idx_at_or_before(ts: np.ndarray, t: int) -> int:
    i = int(np.searchsorted(ts, t, side="right") - 1)
    return max(0, min(i, len(ts) - 1))


def forward_mid_delta(
    ts: np.ndarray, mid: np.ndarray, t_trade: int, k: int
) -> float | None:
    i0 = idx_at_or_before(ts, int(t_trade))
    i1 = min(i0 + int(k), len(ts) - 1)
    if i1 <= i0 and k > 0:
        return None
    return float(mid[i1] - mid[i0])


def cross_forward_extract(
    idx_u: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    day: int,
    ts_sym: np.ndarray,
    t_trade: int,
    k: int,
) -> float | None:
    """Extract mid change from time aligned to VEV's i0 and i0+K timestamps on VEV grid."""
    ukey = (int(day), "VELVETFRUIT_EXTRACT")
    if ukey not in idx_u:
        return None
    ts_u, mid_u, _, _ = idx_u[ukey]
    i0 = idx_at_or_before(ts_sym, int(t_trade))
    i1 = min(i0 + int(k), len(ts_sym) - 1)
    t0 = int(ts_sym[i0])
    t1 = int(ts_sym[i1])
    j0 = idx_at_or_before(ts_u, t0)
    j1 = idx_at_or_before(ts_u, t1)
    return float(mid_u[j1] - mid_u[j0])


def annotate_trades(tr: pd.DataFrame, pr: pd.DataFrame, idx: dict) -> pd.DataFrame:
    rows = []
    for _, r in tr.iterrows():
        d, sym = int(r["day"]), str(r["symbol"])
        t = int(r["timestamp"])
        key = (d, sym)
        if key not in idx:
            continue
        ts, mid, bid, ask = idx[key]
        i0 = idx_at_or_before(ts, t)
        b, a = float(bid[i0]), float(ask[i0])
        px = float(r["price"])
        if np.isnan(b) or np.isnan(a) or a <= b:
            side = "unknown"
        elif px >= a:
            side = "aggr_buy"
        elif px <= b:
            side = "aggr_sell"
        else:
            side = "inside"
        sp = float(a - b) if np.isfinite(a - b) else np.nan
        rows.append(
            {
                **{k: r[k] for k in tr.columns},
                "aggressor_bucket": side,
                "bbo_bid": b,
                "bbo_ask": a,
                "half_spread": sp / 2.0 if np.isfinite(sp) else np.nan,
                "spread": sp,
            }
        )
    return pd.DataFrame(rows)


def _fwd_hydro_at_trade(
    idx: dict, day: int, ts_sym: np.ndarray, t_trade: int, k: int
) -> float | None:
    hk = (int(day), "HYDROGEL_PACK")
    if hk not in idx:
        return None
    ts_h, mid_h, _, _ = idx[hk]
    i0 = idx_at_or_before(ts_sym, int(t_trade))
    i1 = min(i0 + int(k), len(ts_sym) - 1)
    t0, t1 = int(ts_sym[i0]), int(ts_sym[i1])
    j0 = idx_at_or_before(ts_h, t0)
    j1 = idx_at_or_before(ts_h, t1)
    return float(mid_h[j1] - mid_h[j0])


def add_forward_and_burst(tr: pd.DataFrame, idx: dict) -> pd.DataFrame:
    burst_size = tr.groupby(["day", "timestamp"]).size().rename("burst_n")
    tr2 = tr.merge(burst_size, on=["day", "timestamp"], how="left")
    tr2["burst"] = (tr2["burst_n"] >= 2).astype(int)

    recs: list[dict] = []
    for _, r in tr2.iterrows():
        d, sym = int(r["day"]), str(r["symbol"])
        t = int(r["timestamp"])
        key = (d, sym)
        ts, mid, _, _ = idx[key]
        rec = dict(r)
        for k in K_HORIZONS:
            rec[f"fwd_mid_k{k}"] = forward_mid_delta(ts, mid, t, k)
            rec[f"fwd_extract_k{k}"] = cross_forward_extract(idx, d, ts, t, k)
            rec[f"fwd_hydro_k{k}"] = _fwd_hydro_at_trade(idx, d, ts, t, k)
        recs.append(rec)
    return pd.DataFrame(recs)


def spread_quantile_bin(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    try:
        return pd.qcut(s.rank(method="first"), 3, labels=["tight", "mid", "wide"], duplicates="drop")
    except Exception:
        return pd.Series(["mid"] * len(s), index=s.index)


def add_session_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["session_tert"] = "mid"
    for d in df["day"].unique():
        m = df["day"] == d
        ts = df.loc[m, "timestamp"].astype(float)
        if ts.nunique() < 3:
            continue
        q1, q2 = float(ts.quantile(0.33)), float(ts.quantile(0.66))

        def _b(x: float) -> str:
            if x <= q1:
                return "early"
            if x <= q2:
                return "mid"
            return "late"

        df.loc[m, "session_tert"] = ts.map(_b)
    return df


def summarize_cell(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    min_n: int = 30,
) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)[value_col].agg(["count", "mean", "median"]).reset_index()
    g = g[g["count"] >= min_n]
    if len(g) == 0:
        return g
    # t-stat vs 0
    def tstat(sub: pd.Series) -> float:
        x = sub.dropna().astype(float)
        if len(x) < 2:
            return float("nan")
        sd = float(x.std(ddof=1))
        if sd <= 0 or not np.isfinite(sd):
            return float("nan")
        return float(x.mean() / (sd / np.sqrt(len(x))))

    # merge back for t — slow but ok for moderate n
    rows = []
    for _, r in g.iterrows():
        mask = np.ones(len(df), dtype=bool)
        for c in group_cols:
            mask &= (df[c] == r[c]) | (df[c].isna() & pd.isna(r[c]))
        sub = df.loc[mask, value_col].dropna().astype(float)
        rows.append({**r.to_dict(), "t_vs_0": tstat(sub), "frac_pos": float((sub > 0).mean())})
    return pd.DataFrame(rows)


def participant_tables(full: pd.DataFrame) -> None:
    for k in K_HORIZONS:
        col = f"fwd_mid_k{k}"
        parts = []
        for role, name_col in [("buyer", "buyer"), ("seller", "seller")]:
            for U in sorted(full[name_col].dropna().unique()):
                sub = full[full[name_col] == U].copy()
                if len(sub) < 50:
                    continue
                for ag in ["aggr_buy", "aggr_sell", "inside", "unknown"]:
                    s2 = sub[sub["aggressor_bucket"] == ag]
                    if len(s2) < 30:
                        continue
                    x = s2[col].dropna().astype(float)
                    if len(x) < 30:
                        continue
                    tstat = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if x.std(ddof=1) > 0 else np.nan
                    parts.append(
                        {
                            "Mark_role": role,
                            "Mark": U,
                            "aggressor_bucket": ag,
                            "K": k,
                            "n": len(x),
                            "mean_fwd_mid": x.mean(),
                            "median": x.median(),
                            "t_vs_0": tstat,
                            "frac_pos": (x > 0).mean(),
                        }
                    )
        pd.DataFrame(parts).sort_values("n", ascending=False).to_csv(
            OUT / f"r4_p1_participant_fwd_mid_k{k}.csv", index=False
        )


def pair_baseline(full: pd.DataFrame) -> None:
    """Cell means E[fwd|buyer,seller,symbol,burst] and residual."""
    col = "fwd_mid_k20"
    gcols = ["buyer", "seller", "symbol", "burst"]
    cell = (
        full.groupby(gcols)[col]
        .agg(n="count", cell_mean="mean")
        .reset_index()
        .query("n >= 25")
    )
    cell.to_csv(OUT / "r4_p1_baseline_cells_fwd20.csv", index=False)
    m = full.merge(cell, on=gcols, how="left")
    m["residual_fwd20"] = m[col] - m["cell_mean"]
    m.dropna(subset=["residual_fwd20"]).groupby(gcols)["residual_fwd20"].agg(
        n="count", mean_resid="mean", std="std"
    ).reset_index().query("n >= 20").sort_values("mean_resid", key=abs, ascending=False).head(40).to_csv(
        OUT / "r4_p1_residual_top_pairs_fwd20.csv", index=False
    )


def graph_edges(full: pd.DataFrame) -> None:
    edges = full.groupby(["buyer", "seller"]).agg(n=("quantity", "size"), notional=("quantity", "sum")).reset_index()
    edges = edges.sort_values("n", ascending=False)
    edges.to_csv(OUT / "r4_p1_directed_edges_buyer_seller.csv", index=False)
    # 2-hop counts A->B->C
    e = edges.head(200)
    adj = defaultdict(Counter)
    for _, r in e.iterrows():
        adj[str(r["buyer"])][str(r["seller"])] += int(r["n"])
    marks = list(adj.keys())
    chains = []
    for a in marks:
        for b, w1 in adj[a].items():
            for c, w2 in adj[b].items():
                chains.append((a, b, c, w1 * w2))
    pd.DataFrame(chains, columns=["A", "B", "C", "weight_prod"]).sort_values(
        "weight_prod", ascending=False
    ).head(50).to_csv(OUT / "r4_p1_twohop_chain_scores.csv", index=False)


def burst_study(full: pd.DataFrame) -> None:
    col = "fwd_mid_k20"
    ext = full[full["symbol"] == "VELVETFRUIT_EXTRACT"].copy()
    if ext.empty:
        return
    b1 = ext[ext["burst"] == 1][col].dropna()
    b0 = ext[ext["burst"] == 0][col].dropna()
    lines = [
        "Burst event study (VELVETFRUIT_EXTRACT trades, fwd_mid_k20 same symbol)",
        f"burst=1 n={len(b1)} mean={b1.mean():.4f} median={b1.median():.4f}",
        f"burst=0 n={len(b0)} mean={b0.mean():.4f} median={b0.median():.4f}",
    ]
    (OUT / "r4_p1_burst_vs_control_extract.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def adverse_markouts(full: pd.DataFrame) -> None:
    """Who is on the aggressive side when same-symbol fwd is bad for liquidity taker?"""
    col = "fwd_mid_k5"
    sub = full[full["aggressor_bucket"] == "aggr_buy"].dropna(subset=[col])
    # passive seller was counterparty — mark seller
    worst = sub.groupby("seller")[col].agg(n="count", mean_fwd="mean").reset_index().query("n>=40").sort_values(
        "mean_fwd", ascending=True
    )
    worst.to_csv(OUT / "r4_p1_aggr_buy_fwd5_by_seller.csv", index=False)
    sub2 = full[full["aggressor_bucket"] == "aggr_sell"].dropna(subset=[col])
    worst2 = sub2.groupby("buyer")[col].agg(n="count", mean_fwd="mean").reset_index().query("n>=40").sort_values(
        "mean_fwd", ascending=False
    )
    worst2.to_csv(OUT / "r4_p1_aggr_sell_fwd5_by_buyer.csv", index=False)


def stratified_table(full: pd.DataFrame) -> None:
    full = add_session_bins(full)
    full["spread_bin"] = spread_quantile_bin(full["spread"])
    col = "fwd_mid_k20"
    gcols = ["symbol", "aggressor_bucket", "spread_bin", "session_tert", "burst"]
    # aggregate per day for stability check
    rows = []
    for d in DAYS:
        sub = full[full["day"] == d]
        if len(sub) < 100:
            continue
        s = summarize_cell(sub, gcols, col, min_n=20)
        s["day"] = d
        rows.append(s)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(OUT / "r4_p1_stratified_fwd20_by_day.csv", index=False)


def main() -> None:
    pr = load_prices()
    tr = load_trades()
    idx = build_ts_mid_index(pr)
    tr_a = annotate_trades(tr, pr, idx)
    full = add_forward_and_burst(tr_a, idx)
    full = add_session_bins(full)
    full["spread_bin"] = spread_quantile_bin(full["spread"])
    full.to_csv(OUT / "r4_p1_trades_enriched.csv", index=False)

    participant_tables(full)
    pair_baseline(full)
    graph_edges(full)
    burst_study(full)
    adverse_markouts(full)
    stratified_table(full)

    # Per-day stability: Mark 01 aggr_buy on VEV_5300 fwd20
    stab = []
    for d in DAYS:
        sub = full[(full["day"] == d) & (full["buyer"] == "Mark 01") & (full["aggressor_bucket"] == "aggr_buy")]
        sub = sub[sub["symbol"] == "VEV_5300"]
        x = sub["fwd_mid_k20"].dropna()
        if len(x) >= 10:
            stab.append({"day": d, "n": len(x), "mean_fwd20": x.mean(), "symbol": "VEV_5300"})
    pd.DataFrame(stab).to_csv(OUT / "r4_p1_example_stability_mark01_vev5300.csv", index=False)

    summary = {
        "days": DAYS,
        "n_trades": int(len(full)),
        "horizon_definition": "K steps on distinct timestamps per (day,symbol), mid change",
        "outputs": [str(p.relative_to(OUT.parent)) for p in sorted(OUT.glob("*"))],
    }
    (OUT / "r4_p1_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
