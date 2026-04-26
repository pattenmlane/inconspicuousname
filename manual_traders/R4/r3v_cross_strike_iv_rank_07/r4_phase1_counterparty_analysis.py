#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mids (tape-only).

Horizon K: K * 100 in raw timestamp units (price rows step by 100).
Mid at t: exact (day, product, timestamp) match. Forward at t+K*100: exact row or NaN.
hour_cs = (timestamp // 100) // 3600 (contiguous 1-hour buckets from first tick of the day in this tape;
on R4 days 1–3 this field only attains 0,1,2 so wide “session” labels collapse for key marks).

Aggression at trade time: compare trade price to concurrent L1 bid/ask on that symbol.
Session stratification: session_bin in {H00_07, H08_15, H16_23} from hour_cs; written to
r4_p1_participant_forward_by_session.csv (requires n≥10 per cell).
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


def summarize(series: pd.Series) -> dict[str, Any]:
    x = series.dropna()
    n = int(len(x))
    if n < 30:
        return {"n": n, "mean": float("nan"), "t": float("nan"), "pos_frac": float("nan")}
    m = float(x.mean())
    s = float(x.std(ddof=1)) if n > 1 else float("nan")
    tstat = float(m / (s / np.sqrt(n))) if s and s == s and s > 1e-12 else float("nan")
    return {"n": n, "mean": m, "t": tstat, "pos_frac": float((x > 0).mean())}


def main() -> None:
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
        spread = float(r["spread"]) if pd.notna(r["spread"]) else float("nan")
        mid0 = float(r["mid_price"])
        agg = classify_agg(px, bid, ask)
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
    df.to_csv(os.path.join(OUT_DIR, "r4_p1_trade_enriched.csv"), index=False)

    burst_counts = df.groupby(["day", "timestamp"]).size().reset_index(name="n_prints")
    burst_ts = set(
        zip(
            burst_counts.loc[burst_counts["n_prints"] > 1, "day"],
            burst_counts.loc[burst_counts["n_prints"] > 1, "timestamp"],
        )
    )
    df["burst"] = [(int(a), int(b)) in burst_ts for a, b in zip(df["day"], df["timestamp"])]

    participant_rows: list[dict[str, Any]] = []
    for side_key, col_side in [("buy_agg", "buyer"), ("sell_agg", "seller")]:
        sub = df[df["agg"] == side_key]
        marks = sorted({m for m in sub[col_side].unique() if str(m).startswith("Mark")})
        for u in marks:
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
                        st = summarize(gg[col])
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
        marks = sorted({m for m in sub[col_side].unique() if str(m).startswith("Mark")})
        for u in marks:
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
                            st = summarize(g3[col])
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

    sub20 = df[df["agg"].isin(["buy_agg", "sell_agg"])].copy()
    sub20["dm"] = sub20["dm_self_k20"]
    sub20.groupby(["pair", "symbol", "spread_bin"]).agg(n=("dm", "count"), mean_dm=("dm", "mean")).reset_index().to_csv(
        os.path.join(OUT_DIR, "r4_p1_pair_cell_means_k20.csv"), index=False
    )
    pm = sub20.groupby("pair")["dm"].mean()
    sub20["residual"] = sub20["dm"] - sub20["pair"].map(pm)
    pr = (
        sub20.groupby("pair")["residual"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_res", "count": "count"})
    )
    pr = pr[pr["count"] >= 40].sort_values("mean_res", key=np.abs, ascending=False)
    pr.head(30).to_csv(os.path.join(OUT_DIR, "r4_p1_top_residual_pairs_k20.csv"), index=False)

    all_trades.groupby(["buyer", "seller"]).agg(count=("quantity", "count"), notional=("notional", "sum")).reset_index().sort_values(
        "count", ascending=False
    ).to_csv(os.path.join(OUT_DIR, "r4_p1_graph_buyer_seller_edges.csv"), index=False)

    burst_es = []
    for is_b in [True, False]:
        g = df[df["burst"] == is_b]
        st = summarize(g["dm_ex_k20"])
        st["burst"] = is_b
        burst_es.append(st)
    pd.DataFrame(burst_es).to_csv(os.path.join(OUT_DIR, "r4_p1_burst_extract_fwd_k20.csv"), index=False)

    sub20[sub20["seller"] == "Mark 22"].groupby("buyer")["dm_self_k20"].agg(["mean", "count"]).reset_index().sort_values(
        "mean"
    ).to_csv(os.path.join(OUT_DIR, "r4_p1_mark22_seller_markout_by_buyer_k20.csv"), index=False)

    stab = []
    for d in DAYS:
        g = df[(df["day"] == d) & (df["buyer"] == "Mark 01") & (df["agg"] == "buy_agg") & (df["symbol"] == "VEV_5300")]
        stab.append({"day": d, **summarize(g["dm_self_k20"])})
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
