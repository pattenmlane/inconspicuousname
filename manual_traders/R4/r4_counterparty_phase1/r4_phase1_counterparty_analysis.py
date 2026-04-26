#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-aware tape analysis (suggested direction.txt).

Horizon K: **K steps** = K rows forward in the **same (day, symbol)** price tape
(sorted by timestamp); forward mid change = mid(t+K) - mid(t) at the trade row's index.

Aggressive side (when bid1/ask1 available at trade timestamp):
  - price >= ask1  → aggressive **buy** (buyer lifted ask)
  - price <= bid1  → aggressive **sell** (seller hit bid)
  else → **inside** (both sides ambiguous)

Outputs under manual_traders/R4/r4_counterparty_phase1/outputs/ (CSV + summary txt).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
KS = (5, 20, 100)
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]],
]


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    if not frames:
        raise SystemExit("No price files")
    pr = pd.concat(frames, ignore_index=True)
    pr = pr.rename(columns={"product": "symbol"})
    pr["mid"] = pd.to_numeric(pr["mid_price"], errors="coerce")
    pr["bid1"] = pd.to_numeric(pr["bid_price_1"], errors="coerce")
    pr["ask1"] = pd.to_numeric(pr["ask_price_1"], errors="coerce")
    pr["spread"] = pr["ask1"] - pr["bid1"]
    return pr[["day", "timestamp", "symbol", "mid", "bid1", "ask1", "spread"]].dropna(
        subset=["mid", "symbol"]
    )


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["quantity"] = tr["quantity"].astype(int)
    return tr.dropna(subset=["price", "buyer", "seller", "symbol"])


def attach_price_at_trade(pr: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    m = tr.merge(
        pr,
        on=["day", "timestamp", "symbol"],
        how="left",
        suffixes=("", "_px"),
    )
    return m


def forward_mid_deltas(pr: pd.DataFrame) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """For each (day, symbol): timestamps array, mids array (sorted)."""
    out = {}
    for (d, sym), g in pr.groupby(["day", "symbol"], sort=False):
        g = g.sort_values("timestamp")
        out[(d, sym)] = (g["timestamp"].to_numpy(), g["mid"].to_numpy())
    return out


def index_map(ts_arr: np.ndarray) -> dict[int, int]:
    return {int(t): i for i, t in enumerate(ts_arr)}


def enrich_trades(m: pd.DataFrame, series: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for _, r in m.iterrows():
        d, sym = int(r["day"]), str(r["symbol"])
        key = (d, sym)
        if key not in series:
            continue
        ts_arr, mid_arr = series[key]
        t = int(r["timestamp"])
        imap = index_map(ts_arr)
        if t not in imap:
            continue
        i = imap[t]
        mid0 = float(mid_arr[i])
        fwd = {}
        ok = True
        for K in KS:
            j = i + K
            if j >= len(mid_arr):
                ok = False
                break
            fwd[K] = float(mid_arr[j]) - mid0
        if not ok:
            continue
        bid, ask = r.get("bid1"), r.get("ask1")
        prc = float(r["price"])
        if pd.isna(bid) or pd.isna(ask) or ask <= bid:
            aggr = "unknown"
        elif prc >= float(ask):
            aggr = "buy"
        elif prc <= float(bid):
            aggr = "sell"
        else:
            aggr = "inside"
        sp = float(r["spread"]) if pd.notna(r["spread"]) else np.nan
        rows.append({**r.to_dict(), "mid0": mid0, "aggr": aggr, **{f"fwd_{K}": fwd[K] for K in KS}})
    return pd.DataFrame(rows)


def session_bucket(ts: int, day_ts_min: int, day_ts_max: int) -> int:
    if day_ts_max <= day_ts_min:
        return 0
    x = (ts - day_ts_min) / (day_ts_max - day_ts_min)
    return min(3, int(x * 4))


def spread_bucket(sp: float, qs: np.ndarray) -> int:
    if np.isnan(sp):
        return -1
    return int(np.searchsorted(qs, sp, side="right"))


def participant_tables(en: pd.DataFrame, pr: pd.DataFrame, series: dict) -> None:
    """Per Mark U: stratified markout summaries -> CSV."""
    en = en.copy()
    # day-level timestamp quartiles for session
    bounds = {}
    for d in DAYS:
        ts = pr.loc[pr["day"] == d, "timestamp"]
        bounds[d] = (int(ts.min()), int(ts.max()))
    en["sess"] = en.apply(
        lambda r: session_bucket(int(r["timestamp"]), bounds[int(r["day"])][0], bounds[int(r["day"])][1]),
        axis=1,
    )
    # spread quartiles per symbol pooled across days
    qs_by_sym = {}
    for sym in en["symbol"].unique():
        sp = pr.loc[pr["symbol"] == sym, "spread"].dropna().to_numpy()
        if len(sp) < 10:
            qs_by_sym[sym] = np.quantile(sp, [0.25, 0.5, 0.75])
        else:
            qs_by_sym[sym] = np.quantile(sp, [0.25, 0.5, 0.75])
    en["sp_q"] = en.apply(lambda r: spread_bucket(float(r["spread"]), qs_by_sym.get(str(r["symbol"]), np.array([0.0, 1.0, 2.0]))), axis=1)

    names = sorted(set(en["buyer"].astype(str)) | set(en["seller"].astype(str)))
    recs = []
    for U in names:
        sub_b = en[en["buyer"] == U]
        sub_s = en[en["seller"] == U]
        for side, sub in [("as_buyer", sub_b), ("as_seller", sub_s)]:
            if sub.empty:
                continue
            for ag in ["buy", "sell", "inside", "unknown"]:
                a = sub[sub["aggr"] == ag]
                if len(a) < 3:
                    continue
                for sym in a["symbol"].unique():
                    aa = a[a["symbol"] == sym]
                    if len(aa) < 3:
                        continue
                    for K in KS:
                        col = f"fwd_{K}"
                        xx = aa[col].dropna().to_numpy()
                        if len(xx) < 3:
                            continue
                        sd = float(np.std(xx, ddof=1)) if len(xx) > 1 else 0.0
                        tstat = (
                            float(np.mean(xx) / (sd / math.sqrt(len(xx))))
                            if sd > 1e-12
                            else float("nan")
                        )
                        for sess in sorted(aa["sess"].unique()):
                            for spq in sorted(aa["sp_q"].unique()):
                                cell = aa[(aa["sess"] == sess) & (aa["sp_q"] == spq)]
                                zz = cell[col].dropna().to_numpy()
                                if len(zz) < 3:
                                    continue
                                sd2 = float(np.std(zz, ddof=1)) if len(zz) > 1 else 0.0
                                ts2 = (
                                    float(np.mean(zz) / (sd2 / math.sqrt(len(zz))))
                                    if sd2 > 1e-12
                                    else float("nan")
                                )
                                recs.append(
                                    {
                                        "mark": U,
                                        "side": side,
                                        "aggr": ag,
                                        "symbol": sym,
                                        "sess_q": int(sess),
                                        "spread_q_bucket": int(spq),
                                        "K": K,
                                        "n": len(zz),
                                        "mean": float(np.mean(zz)),
                                        "median": float(np.median(zz)),
                                        "frac_pos": float(np.mean(zz > 0)),
                                        "tstat": ts2,
                                    }
                                )
    pd.DataFrame(recs).to_csv(OUT / "r4_p1_participant_markout_by_symbol_K.csv", index=False)

    # Cross-asset: after trade in sym, forward VELVET mid change
    u_series = {d: series[(d, "VELVETFRUIT_EXTRACT")] for d in DAYS if (d, "VELVETFRUIT_EXTRACT") in series}
    u_imap = {d: index_map(u_series[d][0]) for d in u_series}
    cross = []
    for _, r in en.iterrows():
        d = int(r["day"])
        if d not in u_imap:
            continue
        ts_arr, mid_arr = u_series[d]
        im = u_imap[d]
        t = int(r["timestamp"])
        if t not in im:
            continue
        ui = im[t]
        u0 = float(mid_arr[ui])
        for K in KS:
            if ui + K >= len(mid_arr):
                continue
            cross.append(
                {
                    "day": d,
                    "trade_symbol": r["symbol"],
                    "mark_buyer": r["buyer"],
                    "mark_seller": r["seller"],
                    "aggr": r["aggr"],
                    "K": K,
                    "fwd_extract": float(mid_arr[ui + K]) - u0,
                }
            )
    pd.DataFrame(cross).to_csv(OUT / "r4_p1_cross_extract_forward.csv", index=False)


def bot_baseline_residuals(en: pd.DataFrame) -> None:
    en = en.copy()
    en["pair"] = en["buyer"].astype(str) + "->" + en["seller"].astype(str)
    en["regime"] = (pd.cut(en["spread"].clip(0, 50), bins=[0, 2, 6, 200], labels=["tight", "med", "wide"])).astype(str)
    gcols = ["pair", "symbol", "regime"]
    for K in KS:
        col = f"fwd_{K}"
        means = en.groupby(gcols)[col].mean().rename("baseline")
        m2 = en.merge(means, on=gcols, how="left")
        m2["residual"] = m2[col] - m2["baseline"]
        m2.to_csv(OUT / f"r4_p1_residuals_K{K}.csv", index=False)
    # aggregate residual magnitude by pair
    agg = []
    for K in KS:
        col = f"fwd_{K}"
        means = en.groupby(["pair", "symbol", "regime"])[col].mean().rename("baseline")
        m2 = en.merge(means, on=["pair", "symbol", "regime"], how="left")
        m2["residual"] = m2[col] - m2["baseline"]
        for p, g in m2.groupby("pair"):
            agg.append(
                {
                    "pair": p,
                    "K": K,
                    "mean_abs_resid": float(np.mean(np.abs(g["residual"]))),
                    "n": len(g),
                }
            )
    pd.DataFrame(agg).to_csv(OUT / "r4_p1_pair_residual_summary.csv", index=False)


def graph_motifs(en: pd.DataFrame) -> None:
    c = Counter(zip(en["buyer"].astype(str), en["seller"].astype(str)))
    lines = ["=== buyer -> seller edge counts (all days) ===", f"unique pairs: {len(c)}"]
    for (b, s), n in c.most_common(40):
        lines.append(f"{b} -> {s}: {n}")
    (OUT / "r4_p1_graph_top_pairs.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 2-hop counts A->B->C on same day within short time - sample: consecutive trades
    en2 = en.sort_values(["day", "timestamp"])
    hops = Counter()
    for d in DAYS:
        sub = en2[en2["day"] == d]
        prev = None
        for _, r in sub.iterrows():
            if prev is not None and r["timestamp"] - prev["timestamp"] <= 500:
                hops[(prev["buyer"], prev["seller"], r["buyer"], r["seller"])] += 1
            prev = r
    hlines = ["=== 2-trade hops within 500 timestamp units (same day) ==="]
    for k, v in hops.most_common(30):
        hlines.append(f"{k}: {v}")
    (OUT / "r4_p1_graph_2hop_local.txt").write_text("\n".join(hlines) + "\n", encoding="utf-8")


def burst_study(tr: pd.DataFrame, pr: pd.DataFrame, series: dict) -> None:
    g = tr.groupby(["day", "timestamp"]).agg(
        n=("symbol", "count"),
        buyer=("buyer", lambda s: s.mode().iloc[0] if len(s) else ""),
        symbols=("symbol", lambda s: ",".join(sorted(set(s)))),
    )
    bursts = g[g["n"] >= 2].reset_index()
    # forward extract after burst timestamps
    u_key = "VELVETFRUIT_EXTRACT"
    rows = []
    for _, b in bursts.iterrows():
        d, t = int(b["day"]), int(b["timestamp"])
        key = (d, u_key)
        if key not in series:
            continue
        ts_arr, mid_arr = series[key]
        imap = index_map(ts_arr)
        if t not in imap:
            continue
        i = imap[t]
        u0 = float(mid_arr[i])
        for K in KS:
            if i + K < len(mid_arr):
                rows.append(
                    {
                        "day": d,
                        "timestamp": t,
                        "burst_n": int(b["n"]),
                        "orchestrator_buyer": str(b["buyer"]),
                        "fwd_extract": float(mid_arr[i + K]) - u0,
                        "K": K,
                    }
                )
    burst_df = pd.DataFrame(rows)
    # controls: random sample of same-day timestamps with single-trade
    rng = np.random.default_rng(0)
    singles = g[g["n"] == 1].reset_index()
    ctrl = []
    if len(singles) > 0:
        samp = singles.sample(min(500, len(singles)), random_state=0)
        for _, r in samp.iterrows():
            d, t = int(r["day"]), int(r["timestamp"])
            key = (d, u_key)
            if key not in series:
                continue
            ts_arr, mid_arr = series[key]
            imap = index_map(ts_arr)
            if t not in imap:
                continue
            i = imap[t]
            u0 = float(mid_arr[i])
            for K in KS:
                if i + K < len(mid_arr):
                    ctrl.append({"day": d, "fwd_extract": float(mid_arr[i + K]) - u0, "K": K})
    ctrl_df = pd.DataFrame(ctrl)
    burst_df.to_csv(OUT / "r4_p1_burst_forward_extract.csv", index=False)
    ctrl_df.to_csv(OUT / "r4_p1_control_forward_extract.csv", index=False)
    lines = ["=== Burst vs control: forward extract mid change ==="]
    for K in KS:
        a = burst_df[burst_df["K"] == K]["fwd_extract"].dropna()
        b = ctrl_df[ctrl_df["K"] == K]["fwd_extract"].dropna() if len(ctrl_df) else pd.Series(dtype=float)
        if len(a) < 5:
            continue
        bm = float(a.mean())
        cm = float(b.mean()) if len(b) else float("nan")
        lines.append(f"K={K} burst n={len(a)} mean={bm:.5f} | control n={len(b)} mean={cm:.5f}")
    (OUT / "r4_p1_burst_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def adverse_markouts(en: pd.DataFrame) -> None:
    """When aggressor buys at ask, forward same-symbol mid (seller was passive)."""
    rows = []
    sub = en[en["aggr"] == "buy"]
    for _, r in sub.iterrows():
        rows.append(
            {
                "passive_seller": r["seller"],
                "aggressive_buyer": r["buyer"],
                "symbol": r["symbol"],
                "fwd_20": r.get("fwd_20"),
            }
        )
    pd.DataFrame(rows).groupby(["passive_seller", "symbol"])["fwd_20"].agg(["count", "mean"]).reset_index().sort_values(
        "mean"
    ).to_csv(OUT / "r4_p1_adverse_aggrbuy_fwd20_by_passive_seller.csv", index=False)


def write_phase1_summary(en: pd.DataFrame) -> dict:
    """Top candidate edges from tables (heuristic scan)."""
    p = pd.read_csv(OUT / "r4_p1_participant_markout_by_symbol_K.csv")
    # filter n>=20 and |tstat|>2 for story
    p2 = p[(p["n"] >= 20) & (p["tstat"].notna())].copy()
    p2["abs_t"] = p2["tstat"].abs()
    top = p2.sort_values("abs_t", ascending=False).head(15)
    top.to_csv(OUT / "r4_p1_top_tstat_cells.csv", index=False)
    return {
        "n_trades_enriched": int(len(en)),
        "top_tstat_csv": str(OUT / "r4_p1_top_tstat_cells.csv"),
    }


def main() -> None:
    pr = load_prices()
    tr = load_trades()
    series = forward_mid_deltas(pr)
    m = attach_price_at_trade(pr, tr)
    en = enrich_trades(m, series)
    en.to_csv(OUT / "r4_p1_trades_enriched.csv", index=False)
    participant_tables(en, pr, series)
    bot_baseline_residuals(en)
    graph_motifs(en)
    burst_study(tr, pr, series)
    adverse_markouts(en)
    meta = write_phase1_summary(en)
    meta_path = OUT / "r4_phase1_run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote outputs to", OUT)


if __name__ == "__main__":
    main()
