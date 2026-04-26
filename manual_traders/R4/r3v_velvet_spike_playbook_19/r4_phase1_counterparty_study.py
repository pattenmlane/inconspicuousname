#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-aware tape study (Prosperity4Data/ROUND_4 days 1–3).

Horizon K: forward mid change over the next K *rows* in the price tape for the
same (day, product), sorted by timestamp (same convention as round4work scripts).

Outputs under manual_traders/R4/r3v_velvet_spike_playbook_19/analysis_outputs/
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"

DAYS = (1, 2, 3)
KS = (5, 20, 100)
PRODUCTS_FOCUS = [
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def load_prices_day(day: int) -> pd.DataFrame:
    p = DATA / f"prices_round_4_day_{day}.csv"
    df = pd.read_csv(p, sep=";")
    df["day"] = day
    df["mid"] = pd.to_numeric(df["mid_price"], errors="coerce")
    df["bid1"] = pd.to_numeric(df["bid_price_1"], errors="coerce")
    df["ask1"] = pd.to_numeric(df["ask_price_1"], errors="coerce")
    df["spread"] = df["ask1"] - df["bid1"]
    return df[["day", "timestamp", "product", "mid", "bid1", "ask1", "spread"]].dropna(
        subset=["mid", "timestamp"]
    )


def load_trades_day(day: int) -> pd.DataFrame:
    p = DATA / f"trades_round_4_day_{day}.csv"
    df = pd.read_csv(p, sep=";")
    df["day"] = day
    df["product"] = df["symbol"].astype(str)
    return df


def build_mid_index(prices: pd.DataFrame) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """For each (day, product): sorted timestamps, mids, bid1, ask1, spread."""
    out: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for (d, sym), g in prices.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        out[(int(d), str(sym))] = {
            "ts": g["timestamp"].to_numpy(dtype=np.int64),
            "mid": g["mid"].to_numpy(dtype=float),
            "bid1": g["bid1"].to_numpy(dtype=float),
            "ask1": g["ask1"].to_numpy(dtype=float),
            "spread": g["spread"].to_numpy(dtype=float),
        }
    return out


def forward_mid_delta(
    idx: dict[tuple[int, str], dict[str, np.ndarray]],
    day: int,
    symbol: str,
    ts: int,
    k: int,
) -> float:
    key = (day, symbol)
    if key not in idx:
        return float("nan")
    ts_arr = idx[key]["ts"]
    mid_arr = idx[key]["mid"]
    j = int(np.searchsorted(ts_arr, ts, side="left"))
    if j >= len(ts_arr) or ts_arr[j] != ts:
        return float("nan")
    j2 = min(j + k, len(mid_arr) - 1)
    return float(mid_arr[j2] - mid_arr[j])


def aggressor_side(row: pd.Series) -> str:
    """buy_aggr | sell_aggr | unknown"""
    p, bid1, ask1 = float(row["price"]), row["bid1"], row["ask1"]
    if pd.isna(bid1) or pd.isna(ask1):
        return "unknown"
    if p >= ask1:
        return "buy_aggr"
    if p <= bid1:
        return "sell_aggr"
    return "unknown"


def spread_bucket(s: float) -> str:
    if pd.isna(s) or s <= 0:
        return "na"
    if s <= 2:
        return "tight"
    if s <= 6:
        return "mid"
    return "wide"


def hour_bucket(ts: int) -> str:
    # coarse: first third / mid / late of day by timestamp rank proxy — use ts//1e6 not good
    h = (int(ts) // 10000) % 24
    if h < 8:
        return "h0_7"
    if h < 16:
        return "h8_15"
    return "h16_23"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pd.concat([load_prices_day(d) for d in DAYS], ignore_index=True)
    idx = build_mid_index(prices)

    trades = pd.concat([load_trades_day(d) for d in DAYS], ignore_index=True)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")

    # merge book at trade time
    m = trades.merge(
        prices.rename(columns={"spread": "book_spread"}),
        on=["day", "timestamp", "product"],
        how="left",
        suffixes=("", "_px"),
    )
    m["aggr"] = m.apply(aggressor_side, axis=1)
    m["spr_b"] = m["book_spread"].apply(spread_bucket)
    m["hour_b"] = m["timestamp"].apply(hour_bucket)
    m["agg_party"] = np.where(
        m["aggr"] == "buy_aggr",
        m["buyer"].astype(str),
        np.where(m["aggr"] == "sell_aggr", m["seller"].astype(str), ""),
    )
    m["agg_role"] = np.where(m["aggr"] == "buy_aggr", "buyer", np.where(m["aggr"] == "sell_aggr", "seller", "neither"))

    # forward deltas same symbol + extract
    for k in KS:
        m[f"fwd_{k}_sym"] = [
            forward_mid_delta(idx, int(d), str(sym), int(ts), k)
            for d, sym, ts in zip(m["day"], m["product"], m["timestamp"])
        ]
        m[f"fwd_{k}_ex"] = [
            forward_mid_delta(idx, int(d), "VELVETFRUIT_EXTRACT", int(ts), k)
            for d, ts in zip(m["day"], m["timestamp"])
        ]

    m.to_csv(OUT / "r4_trades_with_markout.csv", index=False)

    # --- 1) participant-level summary (Mark U, aggressor side, product, spread bucket) ---
    rows = []
    for (day, u_side, u_name, prod, sprb), g in m.groupby(["day", "agg_role", "agg_party", "product", "spr_b"]):
        if u_side == "neither" or not u_name:
            continue
        for k in KS:
            col = f"fwd_{k}_sym"
            x = g[col].dropna()
            if len(x) < 30:
                continue
            rows.append(
                {
                    "day": int(day),
                    "aggressor_party": str(u_name),
                    "side": u_side,
                    "product": str(prod),
                    "spread_bucket": str(sprb),
                    "K": k,
                    "n": int(len(x)),
                    "mean": float(x.mean()),
                    "median": float(x.median()),
                    "frac_pos": float((x > 0).mean()),
                }
            )

    p1 = pd.DataFrame(rows)
    p1.to_csv(OUT / "r4_participant_markout_by_bucket.csv", index=False)

    # --- baseline cell means (buyer, seller, symbol, spread bucket) ---
    cell = (
        m.groupby(["buyer", "seller", "product", "spr_b"])["fwd_20_sym"]
        .agg(["count", "mean"])
        .reset_index()
    )
    cell.columns = ["buyer", "seller", "product", "spr_b", "cell_n", "cell_mean_fwd20"]
    m2 = m.merge(cell, on=["buyer", "seller", "product", "spr_b"], how="left")
    m2["resid20"] = m2["fwd_20_sym"] - m2["cell_mean_fwd20"]
    m2.to_csv(OUT / "r4_trades_with_baseline_resid.csv", index=False)
    resid_summary = (
        m2.groupby(["buyer", "seller", "product"])
        .agg(resid_mean=("resid20", "mean"), resid_n=("resid20", "count"))
        .reset_index()
        .sort_values("resid_mean", key=np.abs, ascending=False)
        .head(40)
    )
    resid_summary.to_csv(OUT / "r4_top_residual_pairs.csv", index=False)

    # --- graph edges buyer -> seller ---
    edges = (
        m.groupby(["buyer", "seller"])
        .agg(n=("price", "count"), notional=("quantity", "sum"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    edges.to_csv(OUT / "r4_graph_edges_buyer_seller.csv", index=False)

    # --- bursts same (day, timestamp) ---
    burst = (
        m.groupby(["day", "timestamp"])
        .agg(
            n_trades=("product", "count"),
            n_syms=("product", "nunique"),
        )
        .reset_index()
    )
    burst["is_burst"] = burst["n_trades"] >= 3
    burst.to_csv(OUT / "r4_burst_summary_by_timestamp.csv", index=False)

    burst_trades = m.merge(
        burst[burst["is_burst"]][["day", "timestamp"]], on=["day", "timestamp"], how="inner"
    )
    # crude control: single-trade timestamps sample
    singles = burst[burst["n_trades"] == 1][["day", "timestamp"]].sample(min(5000, len(burst[burst["n_trades"] == 1])), random_state=0)
    ctrl_m = m.merge(singles, on=["day", "timestamp"], how="inner")

    def fwd_ex_stats(df: pd.DataFrame, k: int) -> dict[str, float]:
        col = f"fwd_{k}_ex"
        x = df[col].dropna()
        if len(x) < 10:
            return {"n": float(len(x)), "mean": float("nan")}
        return {"n": float(len(x)), "mean": float(x.mean()), "median": float(x.median())}

    burst_study = {f"burst_fwd_ex_{k}": fwd_ex_stats(burst_trades, k) for k in KS}
    burst_study.update({f"control_fwd_ex_{k}": fwd_ex_stats(ctrl_m, k) for k in KS})
    (OUT / "r4_burst_event_study_extract.json").write_text(json.dumps(burst_study, indent=2))

    # --- Mark-level adverse proxy: after print, same-symbol forward (passive hurt) ---
    adv = (
        m[m["aggr"] != "unknown"]
        .groupby(["buyer", "seller", "aggr"])
        .agg(
            n=("fwd_5_sym", "count"),
            m5=("fwd_5_sym", "mean"),
            m20=("fwd_20_sym", "mean"),
        )
        .reset_index()
        .sort_values("m20", ascending=True)
        .head(30)
    )
    adv.to_csv(OUT / "r4_adverse_selection_proxy_worst_fwd20.csv", index=False)

    print("Wrote outputs to", OUT)


if __name__ == "__main__":
    main()
