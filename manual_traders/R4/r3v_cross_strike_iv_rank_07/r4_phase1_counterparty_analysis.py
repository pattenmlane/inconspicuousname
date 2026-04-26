#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape-only).

Horizon K: K * 100 in raw timestamp units (price rows step by 100).
Mid at t: exact (day, product, timestamp) match. Forward at t+K*100: exact row or NaN.
hour_cs = (timestamp // 100) // 3600 (contiguous 1-hour buckets from first tick of the day in this tape;
on R4 days 1–3 this field only attains 0,1,2 so wide “session” labels collapse for key marks).

Aggression at trade time: compare trade price to concurrent L1 bid/ask on that symbol.
Participant loops iterate **every distinct buyer/seller string** on the tape (Round 4 has only Mark names).

Session stratification: session_bin in {H00_07, H08_15, H16_23} from hour_cs; written to
r4_p1_participant_forward_by_session.csv (requires n≥10 per cell).

Participant tables also include median, bootstrap 95% CI on mean (400 resamples, seed 42),
and r4_p1_participant_forward_by_burst.csv (multi_print_burst vs isolated_print).
r4_p1_burst_matched_control_ex_k20.json: paired extract k=20 at burst timestamps vs nearest
same-day isolated price timestamp (control).

r4_p1_participant_cross_forward_stats.csv: at each (participant, side, trade_symbol, spread_bin),
mean/median/t/CI on **extract** and **hydro** forward mids (same horizons).

r4_p1_headline_cells_by_day.csv: per-day n/mean/median for Mark67|extract buy_agg, Mark22|5300 sell_agg,
Mark55|extract sell_agg — self / extract / hydro forwards (t only if day n≥5).
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_outputs")
PRICE_GLOB = "Prosperity4Data/ROUND_4/prices_round_4_day_{d}.csv"
TRADE_GLOB = "Prosperity4Data/ROUND_4/trades_round_4_day_{d}.csv"
DAYS = (1, 2, 3)
K_LIST = (5, 20, 100)


def load_prices(day: int) -> pd.DataFrame:
    p = pd.read_csv(PRICE_GLOB.format(d=day), sep=";")
    p["day"] = day
    p["spread"] = p["ask_price_1"] - p["bid_price_1"]
    return p


def load_trades(day: int) -> pd.DataFrame:
    t = pd.read_csv(TRADE_GLOB.format(d=day), sep=";")
    t["day"] = day
    t["notional"] = t["price"].astype(float) * t["quantity"].astype(float)
    return t


def build_mid_lookup(prices: pd.DataFrame) -> dict[tuple[int, str], dict[int, float]]:
    """(day, product) -> {timestamp: mid_price}"""
    out: dict[tuple[int, str], dict[int, float]] = {}
    for (day, prod), g in prices.groupby(["day", "product"]):
        dct = dict(zip(g["timestamp"].astype(int), g["mid_price"].astype(float)))
        out[(int(day), str(prod))] = dct
    return out


def mid_at(lookup: dict[tuple[int, str], dict[int, float]], day: int, sym: str, ts: int) -> float | None:
    d = lookup.get((day, sym))
    if not d:
        return None
    return d.get(int(ts))


def mid_fwd(lookup: dict[tuple[int, str], dict[int, float]], day: int, sym: str, ts: int, k: int) -> float | None:
    return mid_at(lookup, day, sym, int(ts) + int(k) * 100)


def classify_agg(price: float, bid: float, ask: float) -> str:
    if pd.isna(bid) or pd.isna(ask):
        return "unknown"
    if price >= float(ask):
        return "buy_agg"
    if price <= float(bid):
        return "sell_agg"
    return "passive_mid"


def _tape_names(series: pd.Series) -> list[str]:
    return sorted({str(x) for x in series.dropna().unique() if str(x).strip()})


def summarize(series: pd.Series, rng: np.random.Generator | None = None, n_boot: int = 400) -> dict[str, Any]:
    """Mean, median, t on mean, fraction positive, bootstrap 95% CI on mean (if n>=30)."""
    x = series.dropna().astype(float).values
    n = int(len(x))
    if n < 30:
        return {
            "n": n,
            "mean": float("nan"),
            "median": float("nan"),
            "t": float("nan"),
            "pos_frac": float("nan"),
            "mean_ci95_lo": float("nan"),
            "mean_ci95_hi": float("nan"),
        }
    m = float(np.mean(x))
    med = float(np.median(x))
    s = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    tstat = float(m / (s / np.sqrt(n))) if s and s == s and s > 1e-12 else float("nan")
    pos_frac = float(np.mean(x > 0))
    lo, hi = float("nan"), float("nan")
    if rng is not None and n_boot > 0:
        nb = min(n_boot, 800)
        idx = rng.integers(0, n, size=(nb, n))
        boot_means = np.mean(x[idx], axis=1)
        lo = float(np.quantile(boot_means, 0.025))
        hi = float(np.quantile(boot_means, 0.975))
    return {
        "n": n,
        "mean": m,
        "median": med,
        "t": tstat,
        "pos_frac": pos_frac,
        "mean_ci95_lo": lo,
        "mean_ci95_hi": hi,
    }


def summarize_day(series: pd.Series) -> dict[str, Any]:
    """Per-day slice: always report n/mean/median/pos_frac; t only if n>=5 and std>0."""
    x = series.dropna().astype(float).values
    n = int(len(x))
    if n == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "t": float("nan"), "pos_frac": float("nan")}
    m = float(np.mean(x))
    med = float(np.median(x))
    pos_frac = float(np.mean(x > 0))
    tstat = float("nan")
    if n >= 5:
        s = float(np.std(x, ddof=1))
        if s > 1e-12:
            tstat = float(m / (s / np.sqrt(n)))
    return {"n": n, "mean": m, "median": med, "t": tstat, "pos_frac": pos_frac}


def main() -> None:
    rng = np.random.default_rng(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    all_prices = pd.concat([load_prices(d) for d in DAYS], ignore_index=True)
    all_trades = pd.concat([load_trades(d) for d in DAYS], ignore_index=True)
    lookup = build_mid_lookup(all_prices)

    sq = all_prices.groupby("product")["spread"].quantile([0.33, 0.66]).unstack()
    sq.columns = ["q33", "q66"]

    rows = []
    for _, tr in all_trades.iterrows():
        day = int(tr["day"])
        sym = str(tr["symbol"])
        ts = int(tr["timestamp"])
        buyer = str(tr["buyer"]) if pd.notna(tr["buyer"]) else ""
        seller = str(tr["seller"]) if pd.notna(tr["seller"]) else ""
        px = float(tr["price"])
        prow = all_prices[
            (all_prices["day"] == day)
            & (all_prices["product"] == sym)
            & (all_prices["timestamp"] == ts)
        ]
        if prow.empty:
            continue
        r = prow.iloc[0]
        bid, ask = float(r["bid_price_1"]), float(r["ask_price_1"])
        bv = float(pd.to_numeric(r["bid_volume_1"], errors="coerce") or 0.0) if "bid_volume_1" in r.index else 0.0
        av = float(pd.to_numeric(r["ask_volume_1"], errors="coerce") or 0.0) if "ask_volume_1" in r.index else 0.0
        spread = float(r["spread"]) if pd.notna(r["spread"]) else float("nan")
        mid0 = float(r["mid_price"])
        agg = classify_agg(px, bid, ask)
        micro_minus_mid = np.nan
        if agg == "passive_mid" and bid < ask and (bv + av) > 0:
            micro = (ask * bv + bid * av) / (bv + av)
            micro_minus_mid = float(micro - mid0)
        if sym in sq.index and spread == spread:
            if spread <= float(sq.loc[sym, "q33"]):
                sp_bin = "tight"
            elif spread <= float(sq.loc[sym, "q66"]):
                sp_bin = "mid"
            else:
                sp_bin = "wide"
        else:
            sp_bin = "unk"
        hour = (ts // 100) // 3600
        rec: dict[str, Any] = {
            "day": day,
            "timestamp": ts,
            "symbol": sym,
            "buyer": buyer,
            "seller": seller,
            "pair": f"{buyer}|{seller}",
            "price": px,
            "mid0": mid0,
            "spread": spread,
            "spread_bin": sp_bin,
            "hour_cs": int(hour),
            "agg": agg,
            "notional": float(tr["notional"]),
            "microprice_minus_mid": micro_minus_mid,
        }
        for k in K_LIST:
            m1 = mid_fwd(lookup, day, sym, ts, k)
            rec[f"dm_self_k{k}"] = (m1 - mid0) if m1 is not None else np.nan
            ex0 = mid_at(lookup, day, "VELVETFRUIT_EXTRACT", ts)
            exk = mid_fwd(lookup, day, "VELVETFRUIT_EXTRACT", ts, k)
            rec[f"dm_ex_k{k}"] = (exk - ex0) if ex0 is not None and exk is not None else np.nan
            h0 = mid_at(lookup, day, "HYDROGEL_PACK", ts)
            hk = mid_fwd(lookup, day, "HYDROGEL_PACK", ts, k)
            rec[f"dm_hy_k{k}"] = (hk - h0) if h0 is not None and hk is not None else np.nan
        rows.append(rec)

    df = pd.DataFrame(rows)

    burst_counts = df.groupby(["day", "timestamp"]).size().reset_index(name="n_prints")
    burst_ts = set(
        zip(
            burst_counts.loc[burst_counts["n_prints"] > 1, "day"],
            burst_counts.loc[burst_counts["n_prints"] > 1, "timestamp"],
        )
    )
    df["burst"] = [(int(a), int(b)) in burst_ts for a, b in zip(df["day"], df["timestamp"])]

    # Phase 1 bullet 2: cell-mean baseline E[dm_self_k20 | pair, symbol, spread_bin] and per-print residual
    ag = df["agg"].isin(["buy_agg", "sell_agg"])
    cell_mean = (
        df.loc[ag]
        .groupby(["pair", "symbol", "spread_bin"], observed=True)["dm_self_k20"]
        .mean()
        .rename("cell_mean_dm_self_k20")
        .reset_index()
    )
    df = df.merge(cell_mean, on=["pair", "symbol", "spread_bin"], how="left")
    df["residual_cell_dm_self_k20"] = np.where(
        ag,
        df["dm_self_k20"] - df["cell_mean_dm_self_k20"],
        np.nan,
    )

    df.to_csv(os.path.join(OUT_DIR, "r4_p1_trade_enriched.csv"), index=False)

    # Net flow / volume balance per counterparty name (all prints, not only aggressive)
    names = sorted(set(all_trades["buyer"].dropna().astype(str).unique()) | set(all_trades["seller"].dropna().astype(str).unique()))
    bal_rows: list[dict[str, Any]] = []
    for u in names:
        b = all_trades[all_trades["buyer"] == u]
        s = all_trades[all_trades["seller"] == u]
        qb = float(b["quantity"].astype(float).sum()) if len(b) else 0.0
        qs = float(s["quantity"].astype(float).sum()) if len(s) else 0.0
        bal_rows.append(
            {
                "participant": u,
                "n_prints_as_buyer": int(len(b)),
                "n_prints_as_seller": int(len(s)),
                "qty_buyer": qb,
                "qty_seller": qs,
                "net_signed_qty_buy_minus_sell": qb - qs,
            }
        )
    pd.DataFrame(bal_rows).sort_values("net_signed_qty_buy_minus_sell", ascending=False).to_csv(
        os.path.join(OUT_DIR, "r4_p1_participant_flow_balance.csv"), index=False
    )

    participant_rows: list[dict[str, Any]] = []
    for side_key, col_side in [("buy_agg", "buyer"), ("sell_agg", "seller")]:
        sub = df[df["agg"] == side_key]
        for u in _tape_names(sub[col_side]):
            g = sub[sub[col_side] == u]
            if len(g) < 20:
                continue
            for sym in g["symbol"].unique():
                gs = g[g["symbol"] == sym]
                for spb in ["tight", "mid", "wide", "all"]:
                    if spb == "all":
                        gg = gs
                    else:
                        gg = gs[gs["spread_bin"] == spb]
                    if len(gg) < 10:
                        continue
                    for k in K_LIST:
                        col = f"dm_self_k{k}"
                        st = summarize(gg[col], rng=rng)
                        participant_rows.append(
                            {
                                **st,
                                "mark": u,
                                "side": side_key,
                                "symbol": sym,
                                "spread_bin": spb,
                                "horizon_k": k,
                            }
                        )

    pd.DataFrame(participant_rows).to_csv(
        os.path.join(OUT_DIR, "r4_p1_participant_forward_stats.csv"), index=False
    )

    # Cross-asset forwards at trade (symbol) time: extract and hydro mid changes
    cross_rows: list[dict[str, Any]] = []
    for side_key, col_side in [("buy_agg", "buyer"), ("sell_agg", "seller")]:
        sub = df[df["agg"] == side_key]
        for u in _tape_names(sub[col_side]):
            g = sub[sub[col_side] == u]
            if len(g) < 20:
                continue
            for sym in g["symbol"].unique():
                gs = g[g["symbol"] == sym]
                for spb in ["tight", "mid", "wide", "all"]:
                    if spb == "all":
                        gg = gs
                    else:
                        gg = gs[gs["spread_bin"] == spb]
                    if len(gg) < 30:
                        continue
                    for k in K_LIST:
                        for tgt, col in [
                            ("VELVETFRUIT_EXTRACT", f"dm_ex_k{k}"),
                            ("HYDROGEL_PACK", f"dm_hy_k{k}"),
                        ]:
                            st = summarize(gg[col], rng=rng)
                            cross_rows.append(
                                {
                                    **st,
                                    "participant": u,
                                    "side": side_key,
                                    "trade_symbol": sym,
                                    "spread_bin": spb,
                                    "horizon_k": k,
                                    "forward_target": tgt,
                                }
                            )
    if cross_rows:
        pd.DataFrame(cross_rows).to_csv(
            os.path.join(OUT_DIR, "r4_p1_participant_cross_forward_stats.csv"), index=False
        )

    # Per-day stability for headline Phase-1 cells (multi-day tape requirement)
    day_stab: list[dict[str, Any]] = []
    headline_specs = [
        ("Mark 67", "buy_agg", "buyer", "VELVETFRUIT_EXTRACT", "all"),
        ("Mark 22", "sell_agg", "seller", "VEV_5300", "all"),
        ("Mark 55", "sell_agg", "seller", "VELVETFRUIT_EXTRACT", "all"),
    ]
    for name, agg, col_side, sym, spb in headline_specs:
        sub = df[(df["agg"] == agg) & (df[col_side] == name) & (df["symbol"] == sym)]
        if spb != "all":
            sub = sub[sub["spread_bin"] == spb]
        for d in DAYS:
            for k in K_LIST:
                g = sub[sub["day"] == int(d)]
                for col_label, col in [
                    ("self_mid", f"dm_self_k{k}"),
                    ("extract", f"dm_ex_k{k}"),
                    ("hydro", f"dm_hy_k{k}"),
                ]:
                    st = summarize_day(g[col])
                    day_stab.append(
                        {
                            "cell": f"{name}|{agg}|{sym}|{spb}",
                            "day": int(d),
                            "horizon_k": k,
                            "fwd": col_label,
                            **st,
                        }
                    )
    pd.DataFrame(day_stab).to_csv(os.path.join(OUT_DIR, "r4_p1_headline_cells_by_day.csv"), index=False)

    def _session(h: int) -> str:
        h = int(h)
        if h < 8:
            return "H00_07"
        if h < 16:
            return "H08_15"
        return "H16_23"

    session_rows: list[dict[str, Any]] = []
    for side_key, col_side in [("buy_agg", "buyer"), ("sell_agg", "seller")]:
        sub = df[df["agg"] == side_key].copy()
        sub["session_bin"] = sub["hour_cs"].map(_session)
        for u in _tape_names(sub[col_side]):
            g = sub[sub[col_side] == u]
            if len(g) < 20:
                continue
            for sym in g["symbol"].unique():
                gs = g[g["symbol"] == sym]
                for spb in ["tight", "mid", "wide", "all"]:
                    if spb == "all":
                        gg = gs
                    else:
                        gg = gs[gs["spread_bin"] == spb]
                    if len(gg) < 10:
                        continue
                    for sess in sorted(gg["session_bin"].unique()):
                        g3 = gg[gg["session_bin"] == sess]
                        if len(g3) < 10:
                            continue
                        for k in K_LIST:
                            col = f"dm_self_k{k}"
                            st = summarize(g3[col], rng=rng)
                            session_rows.append(
                                {
                                    **st,
                                    "mark": u,
                                    "side": side_key,
                                    "symbol": sym,
                                    "spread_bin": spb,
                                    "session_bin": sess,
                                    "horizon_k": k,
                                }
                            )
    if session_rows:
        pd.DataFrame(session_rows).to_csv(
            os.path.join(OUT_DIR, "r4_p1_participant_forward_by_session.csv"), index=False
        )

    burst_rows: list[dict[str, Any]] = []
    for side_key, col_side in [("buy_agg", "buyer"), ("sell_agg", "seller")]:
        sub = df[df["agg"] == side_key].copy()
        for u in _tape_names(sub[col_side]):
            g = sub[sub[col_side] == u]
            if len(g) < 20:
                continue
            for sym in g["symbol"].unique():
                gs = g[g["symbol"] == sym]
                for spb in ["tight", "mid", "wide", "all"]:
                    if spb == "all":
                        gg = gs
                    else:
                        gg = gs[gs["spread_bin"] == spb]
                    if len(gg) < 10:
                        continue
                    for burst_label, burst_mask in [
                        ("multi_print_burst", gg["burst"]),
                        ("isolated_print", ~gg["burst"]),
                    ]:
                        g2 = gg[burst_mask]
                        if len(g2) < 10:
                            continue
                        for k in K_LIST:
                            col = f"dm_self_k{k}"
                            st = summarize(g2[col], rng=rng)
                            burst_rows.append(
                                {
                                    **st,
                                    "mark": u,
                                    "side": side_key,
                                    "symbol": sym,
                                    "spread_bin": spb,
                                    "burst_stratum": burst_label,
                                    "horizon_k": k,
                                }
                            )
    if burst_rows:
        pd.DataFrame(burst_rows).to_csv(
            os.path.join(OUT_DIR, "r4_p1_participant_forward_by_burst.csv"), index=False
        )

    sub20 = df[df["agg"].isin(["buy_agg", "sell_agg"])].copy()
    sub20["dm"] = sub20["dm_self_k20"]
    sub20.groupby(["pair", "symbol", "spread_bin"]).agg(n=("dm", "count"), mean_dm=("dm", "mean")).reset_index().to_csv(
        os.path.join(OUT_DIR, "r4_p1_pair_cell_means_k20.csv"), index=False
    )
    # Residual vs cell-mean baseline (pair × symbol × spread_bin), then aggregate mean |residual| by pair
    def _std_ddof1(s: pd.Series) -> float:
        x = s.dropna()
        return float(x.std(ddof=1)) if len(x) > 1 else 0.0

    def _mean_abs(s: pd.Series) -> float:
        return float(np.mean(np.abs(s.dropna().values))) if len(s.dropna()) else 0.0

    cell_res = (
        sub20.groupby(["pair", "symbol", "spread_bin"], observed=True)["residual_cell_dm_self_k20"]
        .agg(n="count", std_res=_std_ddof1, mean_abs_res=_mean_abs)
        .reset_index()
    )
    cell_res = cell_res[cell_res["n"] >= 30].sort_values("std_res", ascending=False)
    cell_res.head(50).to_csv(os.path.join(OUT_DIR, "r4_p1_top_cell_residual_dispersion_k20.csv"), index=False)
    # Same content under legacy filename (Phase-1 gate lists this path): within-cell dispersion after cell-mean baseline
    cell_res.head(50).to_csv(os.path.join(OUT_DIR, "r4_p1_top_residual_pairs_k20.csv"), index=False)

    # Phase 2-style: (buyer|seller pair) × symbol × spread × burst stratum — pooled forward stats
    pair_burst_rows: list[dict[str, Any]] = []
    sub_ag = df[df["agg"].isin(["buy_agg", "sell_agg"])].copy()
    for (pair, sym), grp in sub_ag.groupby(["pair", "symbol"], observed=True):
        for spb in ["all", "tight", "mid", "wide"]:
            if spb == "all":
                g = grp
            else:
                g = grp[grp["spread_bin"] == spb]
            if len(g) < 10:
                continue
            for burst_label, bm in [
                ("multi_print_burst", g["burst"]),
                ("isolated_print", ~g["burst"]),
            ]:
                gg = g[bm]
                if len(gg) < 30:
                    continue
                for k in K_LIST:
                    for fwd_key, col in [
                        ("self_mid", f"dm_self_k{k}"),
                        ("extract", f"dm_ex_k{k}"),
                    ]:
                        st = summarize(gg[col], rng=rng)
                        pair_burst_rows.append(
                            {
                                **st,
                                "pair": pair,
                                "symbol": sym,
                                "spread_bin": spb,
                                "burst_stratum": burst_label,
                                "horizon_k": k,
                                "forward": fwd_key,
                            }
                        )
    if pair_burst_rows:
        pd.DataFrame(pair_burst_rows).to_csv(
            os.path.join(OUT_DIR, "r4_p1_pair_forward_by_burst.csv"), index=False
        )

    all_trades.groupby(["buyer", "seller"]).agg(count=("quantity", "count"), notional=("notional", "sum")).reset_index().sort_values(
        "count", ascending=False
    ).to_csv(os.path.join(OUT_DIR, "r4_p1_graph_buyer_seller_edges.csv"), index=False)

    burst_es = []
    for is_b in [True, False]:
        g = df[df["burst"] == is_b]
        st = summarize(g["dm_ex_k20"], rng=rng)
        st["burst"] = is_b
        burst_es.append(st)
    pd.DataFrame(burst_es).to_csv(os.path.join(OUT_DIR, "r4_p1_burst_extract_fwd_k20.csv"), index=False)

    # Matched-time control: each multi-print (day,ts) vs nearest same-day isolated timestamp extract k=20
    burst_keys = (
        df.loc[df["burst"], ["day", "timestamp"]]
        .drop_duplicates()
        .sort_values(["day", "timestamp"])
        .to_records(index=False)
    )
    iso_by_day: dict[int, np.ndarray] = {}
    for d in DAYS:
        ts_iso = (
            df.loc[(df["day"] == d) & (~df["burst"]), "timestamp"].drop_duplicates().sort_values().astype(int).values
        )
        iso_by_day[int(d)] = ts_iso

    pairs_dm: list[float] = []
    pairs_ctrl: list[float] = []
    for day, ts_b in burst_keys:
        ex0 = mid_at(lookup, int(day), "VELVETFRUIT_EXTRACT", int(ts_b))
        exk = mid_fwd(lookup, int(day), "VELVETFRUIT_EXTRACT", int(ts_b), 20)
        if ex0 is None or exk is None:
            continue
        dm_b = float(exk - ex0)
        arr = iso_by_day.get(int(day))
        if arr is None or len(arr) == 0:
            continue
        pos = int(np.searchsorted(arr, int(ts_b)))
        candidates: list[int] = []
        if pos < len(arr):
            candidates.append(int(arr[pos]))
        if pos > 0:
            candidates.append(int(arr[pos - 1]))
        if not candidates:
            continue
        ts_c = min(candidates, key=lambda t: abs(t - int(ts_b)))
        if ts_c == int(ts_b):
            continue
        c0 = mid_at(lookup, int(day), "VELVETFRUIT_EXTRACT", ts_c)
        ck = mid_fwd(lookup, int(day), "VELVETFRUIT_EXTRACT", ts_c, 20)
        if c0 is None or ck is None:
            continue
        dm_c = float(ck - c0)
        pairs_dm.append(dm_b)
        pairs_ctrl.append(dm_c)

    mc_summary: dict[str, Any] = {"n_burst_timestamps_matched": len(pairs_dm)}
    if len(pairs_dm) >= 30:
        b_arr = np.array(pairs_dm, dtype=float)
        c_arr = np.array(pairs_ctrl, dtype=float)
        diff = b_arr - c_arr
        mc_summary["mean_burst_ex_k20"] = float(np.mean(b_arr))
        mc_summary["mean_control_ex_k20"] = float(np.mean(c_arr))
        mc_summary["mean_diff_burst_minus_control"] = float(np.mean(diff))
        mc_summary["t_diff"] = float(
            np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
        ) if np.std(diff, ddof=1) > 1e-12 else float("nan")
        idx = rng.integers(0, len(diff), size=(400, len(diff)))
        boot = np.mean(diff[idx], axis=1)
        mc_summary["diff_mean_ci95_lo"] = float(np.quantile(boot, 0.025))
        mc_summary["diff_mean_ci95_hi"] = float(np.quantile(boot, 0.975))
    with open(os.path.join(OUT_DIR, "r4_p1_burst_matched_control_ex_k20.json"), "w", encoding="utf-8") as f:
        json.dump(mc_summary, f, indent=2)

    sub20[sub20["seller"] == "Mark 22"].groupby("buyer")["dm_self_k20"].agg(["mean", "count"]).reset_index().sort_values(
        "mean"
    ).to_csv(os.path.join(OUT_DIR, "r4_p1_mark22_seller_markout_by_buyer_k20.csv"), index=False)

    stab = []
    for d in DAYS:
        g = df[(df["day"] == d) & (df["buyer"] == "Mark 01") & (df["agg"] == "buy_agg") & (df["symbol"] == "VEV_5300")]
        stab.append({"day": d, **summarize(g["dm_self_k20"], rng=rng)})
    pd.DataFrame(stab).to_csv(os.path.join(OUT_DIR, "r4_p1_mark01_buy_vev5300_by_day_k20.csv"), index=False)

    # 2-hop: Mark A -> Mark B at t1, then Mark B -> Mark C at t2, 0 < t2-t1 <= 5000; correlate extract fwd from t2, k=20
    chain_effects = []
    for day in DAYS:
        tday = all_trades[all_trades["day"] == day].sort_values("timestamp")
        arr = tday.to_dict("records")
        for i in range(len(arr) - 1):
            t1 = int(arr[i]["timestamp"])
            b1, s1 = str(arr[i]["buyer"]), str(arr[i]["seller"])
            for j in range(i + 1, min(i + 50, len(arr))):
                t2 = int(arr[j]["timestamp"])
                if t2 <= t1 or t2 - t1 > 5000:
                    break
                b2, s2 = str(arr[j]["buyer"]), str(arr[j]["seller"])
                if s1 == b2:  # seller of first is buyer of second
                    ex0 = mid_at(lookup, day, "VELVETFRUIT_EXTRACT", t2)
                    exk = mid_fwd(lookup, day, "VELVETFRUIT_EXTRACT", t2, 20)
                    if ex0 is not None and exk is not None:
                        chain_effects.append(
                            {
                                "day": day,
                                "chain": f"{b1}->{s1}->{s2}",
                                "mid1": b1,
                                "mid2": s1,
                                "end": s2,
                                "dm_ex_k20": exk - ex0,
                            }
                        )
    ce = pd.DataFrame(chain_effects)
    if not ce.empty:
        ce.groupby("chain")["dm_ex_k20"].agg(["mean", "count"]).reset_index().sort_values(
            "count", ascending=False
        ).head(40).to_csv(os.path.join(OUT_DIR, "r4_p1_twohop_extract_fwd_k20.csv"), index=False)

    pt = pd.DataFrame(participant_rows)
    top_cand: list[dict[str, Any]] = []
    if not pt.empty:
        pt2 = pt[np.isfinite(pt["t"]) & (pt["n"] >= 40)].sort_values("t", key=np.abs, ascending=False).head(20)
        top_cand = pt2.to_dict(orient="records")

    summary = {
        "n_trades_input": int(len(all_trades)),
        "n_enriched_rows": int(len(df)),
        "n_burst_timestamps": int(burst_counts["n_prints"].gt(1).sum()),
        "top_tstat_cells": top_cand,
    }
    with open(os.path.join(OUT_DIR, "r4_phase1_machine_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Done", OUT_DIR)


if __name__ == "__main__":
    main()
