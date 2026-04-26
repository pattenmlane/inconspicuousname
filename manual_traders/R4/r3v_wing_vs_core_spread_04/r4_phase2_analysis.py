#!/usr/bin/env python3
"""
Round 4 Phase 2 — orthogonal edges (named-bot + burst + microstructure + lead/lag + regimes).

Reads ROUND_4 prices + trades (days 1–3). Writes under outputs/phase2/.

Run: python3 manual_traders/R4/r3v_wing_vs_core_spread_04/r4_phase2_analysis.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]
CORE = ("VEV_5200", "VEV_5300")
HORIZONS = (5, 20, 100)
BURST_W = 100  # ±W tape units for burst neighbourhood (same units as timestamp step)


def load_prices(day: int) -> pd.DataFrame:
    return pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            x = pd.read_csv(p, sep=";")
            x["tape_day"] = d
            frames.append(x)
    return pd.concat(frames, ignore_index=True)


def mid_panel(px: pd.DataFrame) -> pd.DataFrame:
    m = px.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="last")
    for p in PRODUCTS:
        if p not in m.columns:
            m[p] = np.nan
    return m[PRODUCTS]


def spread_panel(px: pd.DataFrame) -> pd.DataFrame:
    def row_spread(g: pd.DataFrame) -> float:
        try:
            b = float(g["bid_price_1"].iloc[0])
            a = float(g["ask_price_1"].iloc[0])
            return a - b
        except Exception:
            return float("nan")

    s = px.groupby(["timestamp", "product"]).apply(row_spread).rename("spr").reset_index()
    return s.pivot_table(index="timestamp", columns="product", values="spr", aggfunc="last")


def microprice(px: pd.DataFrame, prod: str) -> pd.Series:
    sub = px[px["product"] == prod].set_index("timestamp")
    out = []
    for ts in sorted(sub.index.unique()):
        r = sub.loc[ts].iloc[0] if hasattr(sub.loc[ts], "iloc") else sub.loc[ts]
        try:
            bp, bv = float(r["bid_price_1"]), float(r["bid_volume_1"])
            ap, av = float(r["ask_price_1"]), float(abs(float(r["ask_volume_1"])))
            if bv + av == 0:
                continue
            mp = (ap * bv + bp * av) / (bv + av)
            mid = float(r["mid_price"])
            out.append((int(ts), mp - mid))
        except Exception:
            pass
    return pd.Series(dict(out))


def classify_aggression(row: pd.Series) -> str:
    try:
        p, b, a = float(row["price"]), float(row["bid_price_1"]), float(row["ask_price_1"])
    except Exception:
        return "unknown"
    if p >= a:
        return "buy_aggr"
    if p <= b:
        return "sell_aggr"
    return "inside"


def main() -> None:
    # --- Merge trades with book at timestamp for aggression + joint gate at print time ---
    book_parts = []
    for d in DAYS:
        px = load_prices(d)
        b = px[px["product"].isin(PRODUCTS)].copy()
        b["tape_day"] = d
        book_parts.append(b)
    book = pd.concat(book_parts, ignore_index=True)
    book = book.rename(columns={"product": "symbol"})

    tr = load_trades()
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    m = tr.merge(
        book,
        on=["tape_day", "timestamp", "symbol"],
        how="left",
        suffixes=("", "_b"),
    )
    m["aggression"] = m.apply(classify_aggression, axis=1)
    s52 = book[book["symbol"] == "VEV_5200"].set_index(["tape_day", "timestamp"])
    s53 = book[book["symbol"] == "VEV_5300"].set_index(["tape_day", "timestamp"])

    def joint_tight(row) -> bool:
        try:
            k = (int(row["tape_day"]), int(row["timestamp"]))
            r52 = s52.loc[k]
            r53 = s53.loc[k]
            if hasattr(r52, "iloc"):
                r52 = r52.iloc[0]
            if hasattr(r53, "iloc"):
                r53 = r53.iloc[0]
            sp52 = float(r52["ask_price_1"]) - float(r52["bid_price_1"])
            sp53 = float(r53["ask_price_1"]) - float(r53["bid_price_1"])
            return sp52 <= 2 and sp53 <= 2
        except Exception:
            return False

    m["joint_gate"] = m.apply(joint_tight, axis=1)

    # Mark 01→22 **multi-print** burst: same (day,ts), buyer 01 seller 22, ≥4 rows (matches Phase-1 burst size)
    gburst = (
        m[(m["buyer"] == "Mark 01") & (m["seller"] == "Mark 22")]
        .groupby(["tape_day", "timestamp"])
        .size()
        .reset_index(name="n_prints")
    )
    basket_burst = set(zip(gburst.loc[gburst["n_prints"] >= 4, "tape_day"], gburst.loc[gburst["n_prints"] >= 4, "timestamp"]))

    def near_basket_burst(row) -> bool:
        d, t = int(row["tape_day"]), int(row["timestamp"])
        for dt in range(-BURST_W, BURST_W + 1, 100):
            if (d, t + dt) in basket_burst:
                return True
        return False

    m["near_01_22_basket"] = m.apply(near_basket_burst, axis=1)

    # --- Phase 2.1: pair × burst × gate — forward extract K=20 after print ---
    fwd_rows: list[dict] = []
    for d in DAYS:
        px = load_prices(d)
        mp = mid_panel(px)
        u = mp["VELVETFRUIT_EXTRACT"]
        idx = {int(t): i for i, t in enumerate(mp.index)}
        T = len(mp.index)
        ts_list = list(mp.index.astype(int))

        def fwd_u(ti: int, k: int) -> float:
            j = ti + k
            if j >= T:
                return float("nan")
            a, b = u.iloc[ti], u.iloc[j]
            if pd.isna(a) or pd.isna(b):
                return float("nan")
            return float(b - a)

        sub = m[m["tape_day"] == d]
        for _, r in sub.iterrows():
            ti = idx.get(int(r["timestamp"]))
            if ti is None:
                continue
            pair = f"{r['buyer']}->{r['seller']}"
            for k in (20,):
                fwd_rows.append(
                    {
                        "tape_day": d,
                        "pair": pair,
                        "symbol": r["symbol"],
                        "joint_gate": bool(r["joint_gate"]),
                        "near_01_22_basket": bool(r["near_01_22_basket"]),
                        "burst_n": int(m.loc[(m["tape_day"] == d) & (m["timestamp"] == r["timestamp"])].shape[0]),
                        "k": k,
                        "fwd_extract": fwd_u(ti, k),
                    }
                )
    fwd_df = pd.DataFrame(fwd_rows)
    g = (
        fwd_df.groupby(["pair", "symbol", "joint_gate", "near_01_22_basket"])
        .agg(n=("fwd_extract", "count"), mean=("fwd_extract", "mean"), med=("fwd_extract", "median"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    g.to_csv(OUT / "pair_burst_gate_fwd_extract_k20.csv", index=False)

    # Mean-revert vs trend on VEV_5300 after 01→22 basket burst timestamps
    mr_rows: list[dict] = []
    for d in DAYS:
        px = load_prices(d)
        mp = mid_panel(px)
        v = mp["VEV_5300"]
        idx = {int(t): i for i, t in enumerate(mp.index)}
        T = len(mp.index)
        for ts in sorted(basket_burst):
            if ts[0] != d:
                continue
            ti = idx.get(int(ts[1]))
            if ti is None or ti + 100 >= T:
                continue
            x0 = float(v.iloc[ti]) if not pd.isna(v.iloc[ti]) else float("nan")
            x5 = float(v.iloc[ti + 5]) if ti + 5 < T else float("nan")
            x20 = float(v.iloc[ti + 20]) if ti + 20 < T else float("nan")
            if not math.isnan(x0):
                mr_rows.append({"tape_day": d, "d5": x5 - x0, "d20": x20 - x0})
    mr = pd.DataFrame(mr_rows)
    if len(mr):
        mr.groupby("tape_day").agg(mean5=("d5", "mean"), mean20=("d20", "mean"), n=("d5", "count")).reset_index().to_csv(
            OUT / "vev5300_after_01_22_basket_burst_drift.csv", index=False
        )

    # --- Phase 2.2: microstructure — spread compression -> next |Δmid| extract ---
    comp_rows: list[dict] = []
    for d in DAYS:
        px = load_prices(d)
        sp = spread_panel(px)
        u = mid_panel(px)["VELVETFRUIT_EXTRACT"]
        if "VELVETFRUIT_EXTRACT" not in sp.columns:
            continue
        s = sp["VELVETFRUIT_EXTRACT"].reindex(u.index)
        ds = s.diff()
        du = u.diff().abs()
        for i in range(1, len(u) - 5):
            if pd.isna(ds.iloc[i]) or pd.isna(s.iloc[i - 1]):
                continue
            if s.iloc[i - 1] <= 0:
                continue
            compressed = ds.iloc[i] < 0 and (s.iloc[i] < s.iloc[i - 1])
            if not compressed:
                continue
            fut = float(u.iloc[i + 5] - u.iloc[i]) if not pd.isna(u.iloc[i + 5]) else float("nan")
            fut_abs = abs(fut) if not math.isnan(fut) else float("nan")
            comp_rows.append({"tape_day": d, "ts": int(u.index[i]), "fut_abs_5": fut_abs})
    pd.DataFrame(comp_rows).to_csv(OUT / "extract_spread_compression_fwd_abs5.csv", index=False)

    mp_dev = (
        pd.DataFrame(comp_rows)
        .groupby("tape_day")
        .agg(n=("fut_abs_5", "count"), mean_abs_fwd=("fut_abs_5", "mean"))
        .reset_index()
    )
    mp_dev.to_csv(OUT / "extract_spread_compression_summary_by_day.csv", index=False)

    # --- Phase 2.3: cross-instrument lag corr — signed flow proxy vs Δmid ---
    lag_rows: list[dict] = []
    for d in DAYS:
        px = load_prices(d)
        mp = mid_panel(px)
        trd = m[m["tape_day"] == d].copy()
        trd["signed_qty"] = np.where(trd["aggression"] == "buy_aggr", trd["quantity"], np.where(trd["aggression"] == "sell_aggr", -trd["quantity"], 0))
        for sym in ["VELVETFRUIT_EXTRACT", "VEV_5300", "HYDROGEL_PACK"]:
            if sym not in mp.columns:
                continue
            mid = mp[sym].astype(float)
            dmid = mid.diff().fillna(0.0)
            flow = trd[trd["symbol"] == sym].groupby("timestamp")["signed_qty"].sum().reindex(mid.index).fillna(0.0)
            for lag in (0, 1, 2, 5):
                c = flow.shift(lag).corr(dmid)
                lag_rows.append({"tape_day": d, "symbol": sym, "lag": lag, "corr_flow_to_dmid": float(c) if not pd.isna(c) else float("nan")})
    pd.DataFrame(lag_rows).to_csv(OUT / "signed_flow_lag_corr.csv", index=False)

    # --- Phase 2.5: simple smile proxy — OTM wing IV-ish: (mid / max(u-K,eps)) vs ATM — correlate with Mark 01→22 prints ---
    # Use VEV_5200 mid as S proxy, wing = VEV_6000 mid / max(S-6000,1)
    iv_proxy_rows: list[dict] = []
    for d in DAYS:
        px = load_prices(d)
        mp = mid_panel(px)
        S = mp["VEV_5200"].astype(float)
        w = mp["VEV_6000"].astype(float)
        atm = mp["VEV_5300"].astype(float)
        skew = (w / np.maximum(S - 6000, 1.0)) - (atm / np.maximum(S - 5300, 1.0))
        trd = tr[(tr["tape_day"] == d) & (tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22")]
        ts_set = set(trd["timestamp"].astype(int))
        for ts in mp.index.astype(int):
            iv_proxy_rows.append(
                {
                    "tape_day": d,
                    "timestamp": int(ts),
                    "skew_proxy": float(skew.loc[ts]) if not pd.isna(skew.loc[ts]) else float("nan"),
                    "mark_01_22_print": int(ts in ts_set),
                }
            )
    ivdf = pd.DataFrame(iv_proxy_rows).dropna(subset=["skew_proxy"])
    ivdf.groupby(["tape_day", "mark_01_22_print"]).agg(mean_skew=("skew_proxy", "mean"), n=("skew_proxy", "count")).reset_index().to_csv(
        OUT / "iv_skew_proxy_vs_mark01_22.csv", index=False
    )

    # --- Phase 2.6 / 2.7: Mark22 sell pressure vs concurrent extract mid change ---
    # Day-level: correlation Mark22 sell qty at timestamp vs Δmid extract same step
    day_stats = []
    for d in DAYS:
        px = load_prices(d)
        u = mid_panel(px)["VELVETFRUIT_EXTRACT"]
        du = u.diff()
        trd = tr[tr["tape_day"] == d]
        q22 = trd.assign(sell22=np.where(trd["seller"] == "Mark 22", trd["quantity"], 0)).groupby("timestamp")["sell22"].sum().reindex(u.index).fillna(0)
        c = q22.corr(du)
        day_stats.append({"tape_day": d, "corr_mark22_sell_qty_vs_dextract": float(c) if not pd.isna(c) else float("nan")})
    pd.DataFrame(day_stats).to_csv(OUT / "mark22_sell_pressure_vs_dextract_corr.csv", index=False)

    # --- Signals file for trader_v1: Mark 67 buy-aggr on extract ---
    sig = m[
        (m["symbol"] == "VELVETFRUIT_EXTRACT")
        & (m["buyer"] == "Mark 67")
        & (m["aggression"] == "buy_aggr")
    ][["tape_day", "timestamp"]].drop_duplicates()
    sig_list = [[int(r.tape_day), int(r.timestamp)] for r in sig.itertuples()]
    (OUT / "signals_mark67_buy_aggr_extract.json").write_text(json.dumps(sig_list), encoding="utf-8")

    summary = {
        "n_signals_mark67": len(sig_list),
        "basket_burst_timestamps": len(basket_burst),
        "phase2_outputs": [str(OUT / f) for f in sorted(p.name for p in OUT.iterdir() if p.is_file())],
    }
    (OUT / "phase2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
