#!/usr/bin/env python3
"""
Round 4 Phase 1 — counterparty-conditioned forward mid markouts.

Horizon K: K-th future price row (same day, same product) with timestamp strictly
greater than trade timestamp. K in {5, 20, 100}.

Aggressor: trade price vs last BBO with ts <= trade_ts: price >= ask => buy_aggr;
price <= bid => sell_aggr; else mid_unknown.

Outputs: analysis_outputs/*.csv and r4_phase1_summary.txt

Run from repo root:
  python3 manual_traders/R4/r3v_smile_cubic_spline_logk_12/r4_phase1_counterparty_analysis.py
"""
from __future__ import annotations

import math
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
DAYS = (1, 2, 3)
HORIZONS = (5, 20, 100)
EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"


def load_prices_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    for c in ("bid_price_1", "ask_price_1", "mid_price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_trades_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
    return df


def timeline_for(pr: pd.DataFrame, day: int, sym: str) -> tuple[np.ndarray, ...]:
    sub = pr[(pr["day"] == day) & (pr["product"] == sym)].sort_values("timestamp")
    if sub.empty:
        return tuple(np.array([], dtype=np.int64) for _ in range(4))
    sub = sub.groupby("timestamp", as_index=False).last()
    return (
        sub["timestamp"].to_numpy(np.int64),
        sub["mid_price"].to_numpy(float),
        sub["bid_price_1"].to_numpy(float),
        sub["ask_price_1"].to_numpy(float),
    )


def forward_k(ts_arr: np.ndarray, mid_arr: np.ndarray, t0: int, k: int) -> float | None:
    i = bisect_right(ts_arr, t0)
    j = i + k - 1
    if j >= len(ts_arr):
        return None
    return float(mid_arr[j])


def backward_row(ts_arr, bid_arr, ask_arr, mid_arr, t0):
    if len(ts_arr) == 0:
        return None, None, None
    i = bisect_right(ts_arr, t0) - 1
    if i < 0:
        return None, None, None
    return float(bid_arr[i]), float(ask_arr[i]), float(mid_arr[i])


def aggressor(price: float, bid, ask) -> str:
    if bid is None or ask is None or (isinstance(bid, float) and math.isnan(bid)):
        return "unknown"
    if price >= ask - 1e-9:
        return "buy_aggr"
    if price <= bid + 1e-9:
        return "sell_aggr"
    return "mid_unknown"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # timelines[(day,sym)] = (ts, mid, bid, ask)
    timelines: dict[tuple[int, str], tuple] = {}
    for d in DAYS:
        pr = load_prices_day(d)
        for sym in pr["product"].dropna().unique():
            s = str(sym)
            timelines[(d, s)] = timeline_for(pr, d, s)

    trades = pd.concat([load_trades_day(d) for d in DAYS], ignore_index=True)
    trades = trades.dropna(subset=["symbol", "buyer", "seller", "timestamp"])

    rows: list[dict] = []
    for _, r in trades.iterrows():
        day = int(r["day"])
        sym = str(r["symbol"])
        ts = int(r["timestamp"])
        price = float(r["price"])
        ts_arr, mid_arr, bid_arr, ask_arr = timelines.get((day, sym), (np.array([]),) * 4)
        if len(ts_arr) == 0:
            continue
        bid, ask, mid0 = backward_row(ts_arr, bid_arr, ask_arr, mid_arr, ts)
        ag = aggressor(price, bid, ask)
        spr_sym = int(round(ask - bid)) if (bid is not None and ask is not None) else None
        s52 = timelines.get((day, "VEV_5200"), (np.array([]),) * 4)
        s53 = timelines.get((day, "VEV_5300"), (np.array([]),) * 4)
        sp52 = sp53 = None
        if len(s52[0]):
            b52, a52, _ = backward_row(s52[0], s52[2], s52[3], s52[1], ts)
            if b52 is not None and a52 is not None:
                sp52 = int(round(a52 - b52))
        if len(s53[0]):
            b53, a53, _ = backward_row(s53[0], s53[2], s53[3], s53[1], ts)
            if b53 is not None and a53 is not None:
                sp53 = int(round(a53 - b53))
        joint_tight = (
            sp52 is not None
            and sp53 is not None
            and sp52 <= 2
            and sp53 <= 2
        )

        rec = {
            "day": day,
            "timestamp": ts,
            "symbol": sym,
            "buyer": str(r["buyer"]),
            "seller": str(r["seller"]),
            "price": price,
            "qty": int(r["quantity"]),
            "aggressor": ag,
            "mid0": mid0,
            "hour": (ts // 10000) % 24 if ts >= 10000 else 0,
            "spr5200": sp52,
            "spr5300": sp53,
            "joint_gate_tight": joint_tight,
            "spr_symbol": spr_sym,
        }
        for k in HORIZONS:
            mf = forward_k(ts_arr, mid_arr, ts, k)
            rec[f"mark_{k}_same"] = (mf - mid0) if (mf is not None and mid0 is not None) else np.nan
            te, me, bde, ase = timelines.get((day, EXTRACT), (np.array([]),) * 4)
            if len(te):
                _, _, m0e = backward_row(te, bde, ase, me, ts)
                mfe = forward_k(te, me, ts, k)
                rec[f"mark_{k}_extract"] = (
                    (mfe - m0e) if (mfe is not None and m0e is not None) else np.nan
                )
            else:
                rec[f"mark_{k}_extract"] = np.nan
            th, mh, bdh, ash = timelines.get((day, HYDRO), (np.array([]),) * 4)
            if len(th):
                _, _, m0h = backward_row(th, bdh, ash, mh, ts)
                mfh = forward_k(th, mh, ts, k)
                rec[f"mark_{k}_hydro"] = (
                    (mfh - m0h) if (mfh is not None and m0h is not None) else np.nan
                )
            else:
                rec[f"mark_{k}_hydro"] = np.nan
        rows.append(rec)

    wide = pd.DataFrame(rows)
    wide.to_csv(OUT / "r4_trade_markouts_wide.csv", index=False)

    ss = wide["spr_symbol"].astype(float)
    wide["spr_sym_bin"] = pd.cut(
        ss.fillna(99.0),
        bins=[-0.1, 2, 6, 1e9],
        labels=["tight_0_2", "mid_3_6", "wide_7p"],
    )
    strat = (
        wide.groupby(["joint_gate_tight", "spr_sym_bin", "symbol"], dropna=False)["mark_20_same"]
        .agg(n="count", mean="mean")
        .reset_index()
    )
    strat.to_csv(OUT / "r4_markout_jointgate_spreadbin_symbol_k20.csv", index=False)

    # Party-level long
    party_rows: list[dict] = []
    for _, r in wide.iterrows():
        for role, u in (("buyer", r["buyer"]), ("seller", r["seller"])):
            for k in HORIZONS:
                col = f"mark_{k}_same"
                if pd.isna(r[col]):
                    continue
                party_rows.append(
                    {
                        "day": r["day"],
                        "party": u,
                        "role": role,
                        "aggressor": r["aggressor"],
                        "symbol": r["symbol"],
                        "K": k,
                        "mark_same": float(r[col]),
                        "mark_extract": r.get(f"mark_{k}_extract"),
                        "qty": r["qty"],
                    }
                )
    party_df = pd.DataFrame(party_rows)
    g = (
        party_df.groupby(["party", "role", "aggressor", "symbol", "K"], dropna=False)["mark_same"]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    g["t_like"] = g.apply(
        lambda x: float(x["mean"]) / (float(x["std"]) / math.sqrt(float(x["count"])) + 1e-12)
        if x["count"] >= 30 and pd.notna(x["std"]) and float(x["std"]) > 0
        else float("nan"),
        axis=1,
    )
    g.to_csv(OUT / "r4_markout_party_product_horizon.csv", index=False)

    # Pair x symbol x K
    long_parts = []
    for k in HORIZONS:
        c = f"mark_{k}_same"
        tmp = wide[["day", "timestamp", "buyer", "seller", "symbol", "aggressor", "qty", c]].copy()
        tmp = tmp.rename(columns={c: "mark"})
        tmp["K"] = k
        long_parts.append(tmp)
    long_df = pd.concat(long_parts, ignore_index=True)
    long_df = long_df.dropna(subset=["mark"])
    pair_sum = (
        long_df.groupby(["buyer", "seller", "symbol", "K"])
        .agg(n=("mark", "count"), mean_mark=("mark", "mean"), std_mark=("mark", "std"))
        .reset_index()
    )
    gm = float(pair_sum["mean_mark"].mean())
    pair_sum["residual_vs_global_mean"] = pair_sum["mean_mark"] - gm
    pair_sum.to_csv(OUT / "r4_markout_pair_symbol_horizon.csv", index=False)

    ec = wide.groupby(["buyer", "seller"]).agg(n=("symbol", "count"), lots=("qty", "sum")).reset_index()
    ec.to_csv(OUT / "r4_graph_edges.csv", index=False)

    burst = (
        trades.groupby(["day", "timestamp"])
        .agg(
            n=("symbol", "count"),
            buyer_mode=("buyer", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
            seller_mode=("seller", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
        )
        .reset_index()
    )
    burst_big = burst[burst["n"] >= 4].copy()
    burst_big.to_csv(OUT / "r4_burst_events.csv", index=False)

    burst_fx: list[float] = []
    for _, b in burst_big.iterrows():
        day, ts = int(b["day"]), int(b["timestamp"])
        te, me, bde, ase = timelines.get((day, EXTRACT), (np.array([]),) * 4)
        if len(te) == 0:
            continue
        _, _, m0 = backward_row(te, bde, ase, me, ts)
        mf = forward_k(te, me, ts, 20)
        if m0 is not None and mf is not None:
            burst_fx.append(mf - float(m0))

    np.random.seed(0)
    ctrl: list[float] = []
    for d in DAYS:
        te, me, bde, ase = timelines.get((d, EXTRACT), (np.array([]),) * 4)
        if len(te) < 50:
            continue
        for _ in range(400):
            ts = int(np.random.choice(te[15 : len(te) - 25]))
            _, _, m0 = backward_row(te, bde, ase, me, ts)
            mf = forward_k(te, me, ts, 20)
            if m0 is not None and mf is not None:
                ctrl.append(mf - float(m0))

    burst_txt = [
        f"Burst (>=4 prints same day+ts) count={len(burst_fx)}",
        f"  mean mark_20 VELVET mid: {float(np.mean(burst_fx)):.4f}" if burst_fx else "  no bursts",
        f"Random-ts control n={len(ctrl)} mean: {float(np.mean(ctrl)):.4f}" if ctrl else "  no ctrl",
    ]
    (OUT / "r4_burst_forward_summary.txt").write_text("\n".join(burst_txt) + "\n", encoding="utf-8")

    sub = wide[wide["aggressor"] == "buy_aggr"].copy()
    adv = (
        sub.groupby(["seller", "symbol"])
        .agg(n=("mark_20_same", "count"), mean_m20=("mark_20_same", "mean"))
        .reset_index()
    )
    adv = adv[adv["n"] >= 15].sort_values("mean_m20")
    adv.to_csv(OUT / "r4_adverse_aggressor_summary.csv", index=False)

    lines = ["=== Round 4 Phase 1 summary (automated) ===", ""]
    lines.append("Strongest pair cells (n>=50, K=20, by |mean|):")
    p20 = pair_sum[(pair_sum["K"] == 20) & (pair_sum["n"] >= 50)].copy()
    p20["absm"] = p20["mean_mark"].abs()
    for _, x in p20.sort_values("absm", ascending=False).head(15).iterrows():
        lines.append(
            f"  {x['buyer']}->{x['seller']} {x['symbol']} n={int(x['n'])} mean20={x['mean_mark']:.4f}"
        )
    lines.append("")
    lines.append("Party×role with n>=80, K=20, |mean|>0.5 (same-symbol markout):")
    g20 = g[(g["K"] == 20) & (g["count"] >= 80) & (g["mean"].abs() > 0.5)]
    for _, x in g20.sort_values("mean", key=abs, ascending=False).head(15).iterrows():
        lines.append(
            f"  {x['party']} {x['role']} {x['aggressor']} {x['symbol']} n={int(x['count'])} mean={x['mean']:.4f}"
        )
    (OUT / "r4_phase1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
