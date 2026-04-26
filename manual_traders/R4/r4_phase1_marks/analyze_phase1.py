#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned markouts (tape analysis).

Horizon definition (documented for ping_followup_phases.md):
  **K** is in **price-bar steps**: each step advances the tape by one unique
  ``timestamp`` row (100 clock-units between snapshots in this CSV). So
  ``fwd_K`` = mid(symbol, t+K) - mid(symbol, t) at the same ``day``, where
  ``t`` is the snapshot index matching the trade's ``timestamp``.

Aggressive side: trade ``price`` vs concurrent BBO from ``prices`` at
(``day``, ``timestamp``, ``symbol``): ``buy`` if price >= ask_price_1 - 1e-9,
``sell`` if price <= bid_price_1 + 1e-9, else ``mid``.

Run from repo root:
  python3 manual_traders/R4/r4_phase1_marks/analyze_phase1.py

Writes under ``manual_traders/R4/r4_phase1_marks/outputs/``.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# manual_traders/R4/r4_phase1_marks/analyze_phase1.py -> repo root is parents[3]
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
KS = [5, 20, 100]
SYMS_CORE = [
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
]
ALL_PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
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


def t_stat_welch(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    if va == 0 and vb == 0:
        return float("nan")
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / se)


def load_prices_day(day: int) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    """Return raw df, dict symbol -> mid array aligned to ts_ix, and timestamps array."""
    path = DATA / f"prices_round_4_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    ts = np.sort(df["timestamp"].unique())
    ts_to_i = {int(t): i for i, t in enumerate(ts)}
    mids: dict[str, np.ndarray] = {}
    bids: dict[str, np.ndarray] = {}
    asks: dict[str, np.ndarray] = {}
    for sym in ALL_PRODUCTS:
        sub = df[df["product"] == sym].sort_values("timestamp")
        m = sub.set_index("timestamp")["mid_price"].reindex(ts).astype(float).values
        b1 = sub.set_index("timestamp")["bid_price_1"].reindex(ts).astype(float).values
        a1 = sub.set_index("timestamp")["ask_price_1"].reindex(ts).astype(float).values
        mids[sym] = m
        bids[sym] = b1
        asks[sym] = a1
    meta = pd.DataFrame({"timestamp": ts, "ts_ix": np.arange(len(ts))})
    meta["session_tertile"] = pd.qcut(
        ts, q=3, labels=["early", "mid", "late"], duplicates="drop"
    ).astype(str)
    return df, {"ts": ts, "ts_to_i": ts_to_i, "mids": mids, "bids": bids, "asks": asks, "meta": meta}


def spread_ticks(bid: float, ask: float) -> float:
    if not (np.isfinite(bid) and np.isfinite(ask)) or ask <= bid:
        return float("nan")
    return float(ask - bid)


def build_trade_enriched() -> pd.DataFrame:
    frames = []
    for day in DAYS:
        tp = DATA / f"trades_round_4_day_{day}.csv"
        tr = pd.read_csv(tp, sep=";")
        tr["day"] = day
        frames.append(tr)
    tr = pd.concat(frames, ignore_index=True)
    tr["qty"] = tr["quantity"].astype(int)
    tr["price"] = tr["price"].astype(float)

    sym_to_j = {s: j for j, s in enumerate(ALL_PRODUCTS)}
    out_frames: list[pd.DataFrame] = []

    for day in DAYS:
        _, pack = load_prices_day(day)
        ts_arr = pack["ts"]
        L = len(ts_arr)
        ts_to_i = pack["ts_to_i"]
        mids = pack["mids"]
        bids = pack["bids"]
        asks = pack["asks"]
        meta = pack["meta"]
        tert_map = dict(zip(meta["timestamp"].astype(int), meta["session_tertile"]))

        bid_stk = np.stack([bids[s] for s in ALL_PRODUCTS], axis=0)
        ask_stk = np.stack([asks[s] for s in ALL_PRODUCTS], axis=0)
        mid_stk = np.stack([mids[s] for s in ALL_PRODUCTS], axis=0)

        sub = tr[tr["day"] == day].copy()
        sub["sym"] = sub["symbol"].astype(str)
        sub = sub[sub["sym"].isin(sym_to_j)]
        sub["ts_ix"] = sub["timestamp"].astype(int).map(ts_to_i)
        sub = sub[sub["ts_ix"].notna()].copy()
        sub["ts_ix"] = sub["ts_ix"].astype(int)
        invalid = (sub["ts_ix"] < 0) | (sub["ts_ix"] >= L)
        sub = sub[~invalid]

        j = sub["sym"].map(sym_to_j).astype(int).values
        i = sub["ts_ix"].values
        bid1 = bid_stk[j, i]
        ask1 = ask_stk[j, i]
        mid0 = mid_stk[j, i]
        px = sub["price"].astype(float).values

        side = np.where(
            np.isfinite(ask1) & (px >= ask1 - 1e-9),
            "aggr_buy",
            np.where(np.isfinite(bid1) & (px <= bid1 + 1e-9), "aggr_sell", "at_mid"),
        )
        sp = ask1 - bid1
        sp_bin = np.where(
            ~np.isfinite(sp) | (sp <= 0),
            "unknown",
            np.where(sp <= 2.5, "tight", np.where(sp <= 8, "mid", "wide")),
        )
        ext0 = mid_stk[sym_to_j["VELVETFRUIT_EXTRACT"], i]
        hyd0 = mid_stk[sym_to_j["HYDROGEL_PACK"], i]

        sub["side"] = side
        sub["spread_ticks"] = sp
        sub["spread_bin"] = sp_bin
        sub["session_tertile"] = sub["timestamp"].astype(int).map(tert_map).fillna("unknown")
        sub["mid0"] = mid0
        sub["ext0"] = ext0
        sub["hyd0"] = hyd0

        ok = np.isfinite(mid0)
        sub = sub.loc[sub.index[ok]].copy()
        j = j[ok]
        i = i[ok]
        mid0 = mid0[ok]
        ext0 = ext0[ok]
        hyd0 = hyd0[ok]

        for K in KS:
            jf = i + K
            mask = jf < L
            fwd_same = np.full(len(sub), np.nan)
            fwd_ex = np.full(len(sub), np.nan)
            fwd_hy = np.full(len(sub), np.nan)
            idx = np.nonzero(mask)[0]
            if len(idx):
                fwd_same[idx] = mid_stk[j[idx], jf[idx]] - mid0[idx]
                fwd_ex[idx] = mid_stk[sym_to_j["VELVETFRUIT_EXTRACT"], jf[idx]] - ext0[idx]
                fwd_hy[idx] = mid_stk[sym_to_j["HYDROGEL_PACK"], jf[idx]] - hyd0[idx]
            sub[f"fwd_same_{K}"] = fwd_same
            sub[f"fwd_EXTRACT_{K}"] = fwd_ex
            sub[f"fwd_HYDRO_{K}"] = fwd_hy

        out_frames.append(sub)

    return pd.concat(out_frames, ignore_index=True)


def burst_flags(tr: pd.DataFrame) -> pd.DataFrame:
    g = tr.groupby(["day", "timestamp"]).agg(
        n_prints=("symbol", "size"),
        n_syms=("symbol", "nunique"),
        buyer_burst=("buyer", lambda s: s.nunique() == 1),
        seller_burst=("seller", lambda s: s.nunique() == 1),
        top_buyer=("buyer", lambda s: s.mode().iloc[0] if len(s) else ""),
        top_seller=("seller", lambda s: s.mode().iloc[0] if len(s) else ""),
    )
    g = g.reset_index()
    g["burst_ge4"] = g["n_prints"] >= 4
    g["burst_ge8"] = g["n_prints"] >= 8
    return g


def participant_tables(te: pd.DataFrame) -> None:
    """Aggressor tagging: aggr_buy => buyer is taker; aggr_sell => seller is taker."""
    names = sorted(set(te["buyer"].astype(str)) | set(te["seller"].astype(str)))
    rows = []
    for U in names:
        buy_u = te[(te["side"] == "aggr_buy") & (te["buyer"] == U)]
        sell_u = te[(te["side"] == "aggr_sell") & (te["seller"] == U)]
        for side, sub in (("aggr_buy", buy_u), ("aggr_sell", sell_u)):
            if len(sub) < 30:
                continue
            for sym in sub["symbol"].unique():
                s2 = sub[sub["symbol"] == sym]
                if len(s2) < 20:
                    continue
                mode_sp = s2["spread_bin"].mode()
                sp_ref = str(mode_sp.iloc[0]) if len(mode_sp) else "unknown"
                for K in KS:
                    col = f"fwd_same_{K}"
                    x = s2[col].astype(float).values
                    x = x[np.isfinite(x)]
                    if len(x) < 20:
                        continue
                    ctrl = te[
                        (te["symbol"] == sym)
                        & (te["side"] == side)
                        & (te["spread_bin"] == sp_ref)
                    ][col].astype(float).values
                    ctrl = ctrl[np.isfinite(ctrl)]
                    if len(ctrl) < 50:
                        ctrl = te[(te["symbol"] == sym)][col].astype(float).values
                        ctrl = ctrl[np.isfinite(ctrl)]
                    rows.append(
                        {
                            "mark": U,
                            "side": side,
                            "symbol": sym,
                            "K": K,
                            "n": len(x),
                            "mean_fwd_same": float(np.mean(x)),
                            "median_fwd_same": float(np.median(x)),
                            "mean_fwd_EXTRACT": float(
                                np.nanmean(s2[f"fwd_EXTRACT_{K}"].astype(float).values)
                            ),
                            "mean_fwd_HYDRO": float(
                                np.nanmean(s2[f"fwd_HYDRO_{K}"].astype(float).values)
                            ),
                            "frac_pos_same": float(np.mean(x > 0)),
                            "t_vs_pool": t_stat_welch(x, ctrl[: min(len(ctrl), 5000)]),
                            "days_present": ",".join(
                                str(d) for d in sorted(s2["day"].unique().tolist())
                            ),
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "participant_markout_by_side_symbol_K.csv", index=False)


def stratified_cell_stats(te: pd.DataFrame) -> None:
    rows = []
    for (sym, spb, sess, side), g in te.groupby(
        ["symbol", "spread_bin", "session_tertile", "side"]
    ):
        if len(g) < 15:
            continue
        for K in KS:
            col = f"fwd_same_{K}"
            x = g[col].astype(float).values
            x = x[np.isfinite(x)]
            if len(x) < 10:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "spread_bin": spb,
                    "session_tertile": sess,
                    "side": side,
                    "K": K,
                    "n": len(x),
                    "mean": float(np.mean(x)),
                    "median": float(np.median(x)),
                    "frac_pos": float(np.mean(x > 0)),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "stratified_cell_means.csv", index=False)


def bot_baseline_residuals(te: pd.DataFrame) -> None:
    """Cell mean fwd_same_20 by (buyer, seller, symbol, spread_bin); residual on each print."""
    K = 20
    c = f"fwd_same_{K}"

    te = te.copy()
    baseline = {}
    for key, g in te.groupby(["buyer", "seller", "symbol", "spread_bin"]):
        x = g[c].astype(float).values
        x = x[np.isfinite(x)]
        if len(x) >= 5:
            baseline[key] = float(np.mean(x))

    res = []
    for _, r in te.iterrows():
        sym = str(r["symbol"])
        if c not in r or not np.isfinite(r[c]):
            continue
        key = (str(r["buyer"]), str(r["seller"]), sym, str(r["spread_bin"]))
        b = baseline.get(key, float("nan"))
        res.append(
            {
                "residual": float(r[c] - b) if np.isfinite(b) else float("nan"),
                "actual": float(r[c]),
                "baseline": b,
                "buyer": key[0],
                "seller": key[1],
                "symbol": sym,
                "day": int(r["day"]),
            }
        )
    rf = pd.DataFrame(res)
    rf.to_csv(OUT / "baseline_fwd20_residuals.csv", index=False)
    summ = (
        rf.groupby(["buyer", "seller", "symbol"])
        .agg(n=("residual", "size"), mean_res=("residual", "mean"), std_res=("residual", "std"))
        .reset_index()
    )
    summ = summ[summ["n"] >= 8].sort_values("mean_res", key=np.abs, ascending=False)
    summ.head(40).to_csv(OUT / "top_abs_residual_pairs.csv", index=False)


def graph_pairs_notional(te: pd.DataFrame) -> None:
    te["notional"] = te["price"] * te["qty"]
    edges = (
        te.groupby(["buyer", "seller"])
        .agg(n=("symbol", "size"), notional=("notional", "sum"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    edges.to_csv(OUT / "directed_pair_counts_notional.csv", index=False)

    # Sequential two-hop on tape order: (buyer_i, seller_i) then next row's buyer/seller
    chains = Counter()
    sub = te.sort_values(["day", "timestamp", "symbol"])
    b = sub["buyer"].astype(str).values
    s = sub["seller"].astype(str).values
    for i in range(len(sub) - 1):
        chains[(b[i], s[i], b[i + 1])] += 1
        chains[(b[i], s[i], s[i + 1])] += 1
    top = chains.most_common(30)
    with open(OUT / "two_step_chain_counts.txt", "w", encoding="utf-8") as f:
        for (a, b, c), n in top:
            f.write(f"{a} -> {b} | next->{c}: {n}\n")


def burst_event_study(te: pd.DataFrame, tr_raw: pd.DataFrame) -> None:
    bf = burst_flags(tr_raw)
    merged = te.merge(
        bf[
            [
                "day",
                "timestamp",
                "burst_ge4",
                "burst_ge8",
                "n_prints",
                "top_buyer",
                "top_seller",
            ]
        ],
        on=["day", "timestamp"],
        how="left",
    )
    K = 20
    rows = []
    for burst_col, label in [("burst_ge4", "ge4"), ("burst_ge8", "ge8")]:
        for flag in (True, False):
            g = merged[merged[burst_col] == flag]
            x = g[f"fwd_EXTRACT_{K}"].astype(float).values
            x = x[np.isfinite(x)]
            if len(x) < 10:
                continue
            rows.append(
                {
                    "burst_def": label,
                    "is_burst": flag,
                    "K": K,
                    "n": len(x),
                    "mean_fwd_extract": float(np.mean(x)),
                    "median": float(np.median(x)),
                    "frac_pos": float(np.mean(x > 0)),
                }
            )
    df_burst = pd.DataFrame(rows)
    df_burst.to_csv(OUT / "burst_vs_extract_fwd20.csv", index=False)

    # Welch: burst>=4 vs <4 on same merged frame (per-print extract fwd20)
    cnt = tr_raw.groupby(["day", "timestamp"]).size().reset_index(name="burst_n")
    m = te.merge(cnt, on=["day", "timestamp"], how="left")
    x1 = m.loc[m["burst_n"] >= 4, "fwd_EXTRACT_20"].astype(float).dropna().values
    x0 = m.loc[m["burst_n"] < 4, "fwd_EXTRACT_20"].astype(float).dropna().values
    if len(x1) >= 10 and len(x0) >= 10:
        t_b = t_stat_welch(x1, x0)
        with open(OUT / "burst_extract_welch_ge4_vs_lt4.txt", "w", encoding="utf-8") as f:
            f.write(
                f"fwd_EXTRACT_20: burst_n>=4 n={len(x1)} mean={float(np.mean(x1)):.6f}\n"
                f"                burst_n<4  n={len(x0)} mean={float(np.mean(x0)):.6f}\n"
                f"Welch t (>=4 minus <4): {t_b:.4f}\n"
            )


def lagged_flow_extract(te: pd.DataFrame) -> None:
    """Per day: sum signed aggr_buy qty - aggr_sell qty by Mark in rolling time windows — correlate with extract fwd."""
    rows = []
    for day in DAYS:
        sub = te[te["day"] == day].sort_values("timestamp")
        ts_u = sub["timestamp"].values
        signed = np.where(
            sub["side"].values == "aggr_buy",
            sub["qty"].values,
            np.where(sub["side"].values == "aggr_sell", -sub["qty"].values, 0),
        )
        buyer = sub["buyer"].astype(str).values
        marks = sorted(set(buyer.tolist()))
        for M in marks:
            mflow = np.where(buyer == M, signed, 0.0)
            win = 5
            roll = np.convolve(mflow, np.ones(win), mode="valid")
            ext_fwd = sub["fwd_EXTRACT_20"].astype(float).values[win - 1 :]
            mlen = min(len(roll), len(ext_fwd))
            roll, ext_fwd = roll[:mlen], ext_fwd[:mlen]
            mask = np.isfinite(ext_fwd)
            if mask.sum() < 30:
                continue
            corr = np.corrcoef(roll[mask], ext_fwd[mask])[0, 1]
            rows.append({"day": day, "mark": M, "corr_roll5_signed_qty_ext_fwd20": float(corr)})
    pd.DataFrame(rows).to_csv(OUT / "lagged_signed_flow_extract_corr.csv", index=False)


def passive_markout_proxy(te: pd.DataFrame) -> None:
    """When U is buyer (lift ask), passive seller is 'Mark X'; markout for seller perspective: -fwd if price rose."""
    K = 20
    rows = []
    for role in ("buyer", "seller"):
        col_party = role
        for U in te[col_party].astype(str).unique():
            sub = te[te[col_party] == U]
            if len(sub) < 20:
                continue
            sym = "VELVETFRUIT_EXTRACT"
            s2 = sub[sub["symbol"] == sym]
            if len(s2) < 10:
                continue
            fc = f"fwd_same_{K}"
            x = s2[fc].astype(float).values
            x = x[np.isfinite(x)]
            rows.append(
                {
                    "party": U,
                    "role_on_tape": role,
                    "symbol": sym,
                    "n": len(x),
                    "mean_fwd20": float(np.mean(x)),
                    "frac_pos": float(np.mean(x > 0)),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "extract_passive_party_fwd20_proxy.csv", index=False)


def write_summary_md(te: pd.DataFrame) -> None:
    lines = [
        "# Round 4 Phase 1 — summary (automated)",
        "",
        "## Horizon",
        "- **K ∈ {5,20,100}** = forward **price snapshot** steps (unique timestamps per day, +100 per step).",
        "",
        "## Participant predictiveness (high level)",
        "- See `participant_markout_by_side_symbol_K.csv`.",
        "- Flag cells with **|t_vs_pool| > 2** and **n≥50** per day-pool for manual review.",
        "",
        "## Burst vs extract",
        "- See `burst_vs_extract_fwd20.csv`.",
        "",
        "## Baseline residuals",
        "- `baseline_fwd20_residuals.csv` + `top_abs_residual_pairs.csv`.",
        "",
        "## Graph",
        "- `directed_pair_counts_notional.csv`, `two_step_chain_counts.txt`.",
        "",
        f"## Trade prints enriched: **{len(te)}** rows (matched to price grid).",
        "",
    ]
    (OUT / "PHASE1_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Loading trades + building enriched table (may take ~1–2 min)...", flush=True)
    tr_raw = pd.concat(
        [
            pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d)
            for d in DAYS
        ],
        ignore_index=True,
    )
    te = build_trade_enriched()
    te.to_csv(OUT / "trades_enriched.csv", index=False)
    print(f"Enriched trades: {len(te)}", flush=True)

    participant_tables(te)
    stratified_cell_stats(te)
    bot_baseline_residuals(te)
    graph_pairs_notional(te)
    burst_event_study(te, tr_raw)
    lagged_flow_extract(te)
    passive_markout_proxy(te)
    write_summary_md(te)

    sub = te[
        (te["buyer"] == "Mark 01")
        & (te["side"] == "aggr_buy")
        & (te["symbol"] == "VEV_5300")
    ]
    if len(sub):
        x = sub["fwd_same_20"].astype(float).dropna()
        print(
            f"Example cell Mark01 aggr_buy VEV_5300 fwd_same_20: n={len(x)} mean={x.mean():.4f}",
            flush=True,
        )

    print(f"Done. Outputs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
