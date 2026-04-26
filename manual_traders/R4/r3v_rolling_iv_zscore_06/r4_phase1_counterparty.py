#!/usr/bin/env python3
"""Round 4 Phase 1 — counterparty + forward mids. See suggested direction.txt."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
KS = [5, 20, 100]
PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def forward_delta(ts: np.ndarray, mid: np.ndarray, t0: int, k: int) -> float:
    i = int(np.searchsorted(ts, t0, side="left"))
    if i >= len(ts):
        return float("nan")
    if ts[i] != t0:
        i = int(np.searchsorted(ts, t0, side="right") - 1)
    i = max(0, min(i, len(ts) - 1))
    j = min(i + k, len(mid) - 1)
    if j <= i:
        return float("nan")
    return float(mid[j] - mid[i])


def classify_aggr(bid1: float, ask1: float, price: float) -> str:
    if price >= ask1 - 1e-9:
        return "buy_aggr"
    if price <= bid1 + 1e-9:
        return "sell_aggr"
    return "ambiguous"


def main() -> None:
    px = load_prices()
    tr = load_trades()
    tr["qty"] = tr["quantity"].astype(int)

    # Exact join trade -> BBO at same (day, ts, symbol)
    px2 = px.rename(columns={"product": "symbol"})
    fixed = []
    for d in DAYS:
        for sym in PRODUCTS:
            tsub = tr[(tr["day"] == d) & (tr["symbol"] == sym)].sort_values("timestamp")
            psub = px2[(px2["day"] == d) & (px2["symbol"] == sym)].sort_values("timestamp")
            if tsub.empty or psub.empty:
                continue
            mm = pd.merge_asof(
                tsub,
                psub[["timestamp", "bid_price_1", "ask_price_1", "mid_price"]],
                on="timestamp",
                direction="backward",
            )
            mm["day"] = d
            mm["symbol"] = sym
            fixed.append(mm)
    mtr = pd.concat(fixed, ignore_index=True)

    mtr["aggr"] = [
        classify_aggr(float(b), float(a), float(p))
        for b, a, p in zip(mtr["bid_price_1"], mtr["ask_price_1"], mtr["price"])
    ]

    # Series per (day, sym)
    series: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for (d, sym), g in px2.groupby(["day", "symbol"]):
        g = g.sort_values("timestamp")
        series[(int(d), str(sym))] = (
            g["timestamp"].to_numpy(dtype=np.int64),
            g["mid_price"].astype(float).to_numpy(),
        )

    for K in KS:
        same, ext = [], []
        for _, r in mtr.iterrows():
            d, sym, t0 = int(r["day"]), str(r["symbol"]), int(r["timestamp"])
            ts, mid = series[(d, sym)]
            same.append(forward_delta(ts, mid, t0, K))
            ts2, mid2 = series[(d, "VELVETFRUIT_EXTRACT")]
            ext.append(forward_delta(ts2, mid2, t0, K))
        mtr[f"fwd_same_K{K}"] = same
        mtr[f"fwd_ext_K{K}"] = ext

    # --- 1) Participant tables (aggregate over symbols, by role x aggr) ---
    rows = []
    names = sorted(set(mtr["buyer"].astype(str)) | set(mtr["seller"].astype(str)))
    for U in names:
        for role, col in (("as_buyer", "buyer"), ("as_seller", "seller")):
            for ag in ("buy_aggr", "sell_aggr", "all"):
                sub = mtr[mtr[col] == U]
                if ag != "all":
                    sub = sub[sub["aggr"] == ag]
                if len(sub) < 25:
                    continue
                for K in KS:
                    coln = f"fwd_same_K{K}"
                    x = sub[coln].dropna().astype(float)
                    if len(x) < 25:
                        continue
                    tstat = (
                        float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
                        if len(x) > 1 and x.std(ddof=1) > 0
                        else float("nan")
                    )
                    rows.append(
                        {
                            "Mark": U,
                            "role": role,
                            "aggr": ag,
                            "K": K,
                            "n": len(x),
                            "mean_same": x.mean(),
                            "median_same": x.median(),
                            "frac_pos_same": (x > 0).mean(),
                            "t_stat_same": tstat,
                        }
                    )
        for sym in ["VEV_5300", "VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"]:
            sub = mtr[(mtr["buyer"] == U) | (mtr["seller"] == U)]
            sub = sub[sub["symbol"] == sym]
            if len(sub) < 20:
                continue
            for K in KS:
                coln = f"fwd_same_K{K}"
                x = sub[coln].dropna().astype(float)
                if len(x) < 15:
                    continue
                rows.append(
                    {
                        "Mark": U,
                        "role": "either_symbol",
                        "aggr": "all",
                        "K": K,
                        "symbol_filter": sym,
                        "n": len(x),
                        "mean_same": x.mean(),
                        "median_same": x.median(),
                        "frac_pos_same": (x > 0).mean(),
                        "t_stat_same": float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
                        if len(x) > 1 and x.std(ddof=1) > 0
                        else float("nan"),
                    }
                )

    p1 = pd.DataFrame(rows)
    p1.to_csv(OUT / "r4_p1_forward_markout_by_mark.csv", index=False)

    # Day-stability: Mark 01 buy_aggr on VEV_5300 fwd20
    stab = []
    for d in DAYS:
        sub = mtr[(mtr["day"] == d) & (mtr["symbol"] == "VEV_5300") & (mtr["buyer"] == "Mark 01") & (mtr["aggr"] == "buy_aggr")]
        x = sub["fwd_same_K20"].dropna()
        stab.append({"day": d, "n": len(x), "mean_fwd20": x.mean() if len(x) else np.nan})
    pd.DataFrame(stab).to_csv(OUT / "r4_p1_mark01_vev5300_buyaggr_by_day.csv", index=False)

    # --- 2) Baseline + residual ---
    cell = mtr.groupby(["buyer", "seller", "symbol"])["fwd_same_K20"].agg(["mean", "count"]).reset_index()
    cell = cell[cell["count"] >= 5]
    m2 = mtr.merge(cell.rename(columns={"mean": "baseline_fwd20", "count": "cell_n"}), on=["buyer", "seller", "symbol"])
    m2["residual20"] = m2["fwd_same_K20"] - m2["baseline_fwd20"]
    rs = (
        m2.groupby(["buyer", "seller", "symbol"])["residual20"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    rs = rs[rs["count"] >= 8].assign(absmean=lambda d: d["mean"].abs()).sort_values("absmean", ascending=False).head(50)
    rs.drop(columns=["absmean"]).to_csv(OUT / "r4_p1_residual_top50.csv", index=False)

    # --- 3) Pairs + 2-hop ---
    pairs = (
        mtr.assign(notional=mtr["price"].astype(float) * mtr["qty"])
        .groupby(["buyer", "seller"])
        .agg(rows=("timestamp", "size"), notional=("notional", "sum"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    pairs.to_csv(OUT / "r4_p1_graph_pairs.csv", index=False)

    hops = Counter()
    for d in DAYS:
        t = mtr[mtr["day"] == d].sort_values("timestamp")
        prev = None
        for _, r in t.iterrows():
            if prev is not None:
                hops[(str(prev["seller"]), str(r["buyer"]), str(r["seller"]))] += 1
            prev = r
    with open(OUT / "r4_p1_twohop_counts.txt", "w") as f:
        for k, n in hops.most_common(30):
            f.write(f"{k[0]} -> {k[1]} -> {k[2]}: {n}\n")

    # --- 4) Bursts ---
    burst = tr.groupby(["day", "timestamp"]).size().reset_index(name="n_prints")
    burst_big = burst[burst["n_prints"] >= 3]
    burst_rows = []
    for _, b in burst_big.iterrows():
        d, ts = int(b["day"]), int(b["timestamp"])
        sub = mtr[(mtr["day"] == d) & (mtr["timestamp"] == ts)]
        if sub.empty:
            continue
        mode_buy = sub["buyer"].value_counts().index[0]
        ts_e, mid_e = series[(d, "VELVETFRUIT_EXTRACT")]
        burst_rows.append(
            {
                "day": d,
                "timestamp": ts,
                "n_prints": len(sub),
                "mode_buyer": mode_buy,
                "ext_fwd20": forward_delta(ts_e, mid_e, ts, 20),
            }
        )
    bdf = pd.DataFrame(burst_rows)
    bdf.to_csv(OUT / "r4_p1_bursts_rows.csv", index=False)
    ctrl = mtr.sample(min(800, len(mtr)), random_state=1)
    ctrl_e = []
    for _, r in ctrl.iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        ts_e, mid_e = series[(d, "VELVETFRUIT_EXTRACT")]
        ctrl_e.append(forward_delta(ts_e, mid_e, ts, 20))
    with open(OUT / "r4_p1_bursts_eventstudy.txt", "w") as f:
        x = bdf["ext_fwd20"].dropna()
        y = pd.Series(ctrl_e).dropna()
        f.write(f"burst_rows={len(x)} mean_ext_fwd20={x.mean():.6g} std={x.std():.6g}\n")
        f.write(f"control_n={len(y)} mean_ext_fwd20={y.mean():.6g}\n")

    # --- 5) Adverse: aggressive buy, markout by seller ---
    ba = mtr[mtr["aggr"] == "buy_aggr"]
    adv = ba.groupby("seller")["fwd_same_K20"].agg(["mean", "count"]).reset_index()
    adv = adv[adv["count"] >= 10].sort_values("mean")
    adv.to_csv(OUT / "r4_p1_aggressive_buy_markout_by_seller.csv", index=False)

    top = p1[(p1["n"] >= 40) & (p1["t_stat_same"].abs() > 1.96)].sort_values("t_stat_same", key=abs, ascending=False).head(25)
    top.to_csv(OUT / "r4_p1_candidate_edges_t196.csv", index=False)

    # Hour bucket (timestamp // 1e6 as coarse proxy — ticks are small ints ~4500-1e7)
    mtr["hour_bin"] = (mtr["timestamp"] // 1_000_000).astype(int)
    hb = (
        mtr[mtr["symbol"] == "VEV_5300"]
        .groupby(["hour_bin", "buyer", "aggr"])["fwd_same_K20"]
        .agg(["mean", "count"])
        .reset_index()
    )
    hb = hb[hb["count"] >= 15]
    hb.to_csv(OUT / "r4_p1_vev5300_by_hourbin_buyer.csv", index=False)

    # Day-stability slices (raw prints)
    stab67 = []
    for d in DAYS:
        sub = mtr[(mtr["day"] == d) & (mtr["buyer"] == "Mark 67") & (mtr["aggr"] == "buy_aggr")]
        x = sub["fwd_same_K5"].dropna()
        stab67.append({"slice": "Mark67_buy_aggr_K5_same", "day": d, "n": len(x), "mean": x.mean(), "frac_pos": (x > 0).mean() if len(x) else np.nan})
    pd.DataFrame(stab67).to_csv(OUT / "r4_p1_stability_mark67_buyaggr_k5_by_day.csv", index=False)

    stab22 = []
    for d in DAYS:
        sub = mtr[(mtr["day"] == d) & (mtr["seller"] == "Mark 22") & (mtr["aggr"] == "buy_aggr")]
        x = sub["fwd_same_K5"].dropna()
        stab22.append({"slice": "vsMark22_buy_aggr_K5_same", "day": d, "n": len(x), "mean": x.mean(), "frac_pos": (x > 0).mean() if len(x) else np.nan})
    pd.DataFrame(stab22).to_csv(OUT / "r4_p1_stability_seller22_buyaggr_k5_by_day.csv", index=False)

    print("done", OUT, "rows", len(mtr))


if __name__ == "__main__":
    main()
