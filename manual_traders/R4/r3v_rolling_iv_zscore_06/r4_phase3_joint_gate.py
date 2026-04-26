#!/usr/bin/env python3
"""Round 4 Phase 3 — Sonic joint gate on Mark67 aggressive extract: forward extract mid K=5."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
TH = 2
DAYS = [1, 2, 3]


def load_px() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def load_tr() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def series_pair(px: pd.DataFrame, day: int, sym: str) -> tuple[np.ndarray, np.ndarray]:
    g = px[(px["day"] == day) & (px["product"] == sym)].sort_values("timestamp")
    return g["timestamp"].to_numpy(dtype=np.int64), g["mid_price"].astype(float).to_numpy()


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


def main() -> None:
    px = load_px()
    tr = load_tr()
    sp = px[px["product"].isin(["VEV_5200", "VEV_5300"])].copy()
    sp["spread"] = sp["ask_price_1"] - sp["bid_price_1"]
    sp52 = sp[sp["product"] == "VEV_5200"][["day", "timestamp", "spread"]].rename(columns={"spread": "s52"})
    sp53 = sp[sp["product"] == "VEV_5300"][["day", "timestamp", "spread"]].rename(columns={"spread": "s53"})
    jt = sp52.merge(sp53, on=["day", "timestamp"])
    jt["joint_tight"] = (jt["s52"] <= TH) & (jt["s53"] <= TH)
    jmap = jt.set_index(["day", "timestamp"])["joint_tight"].to_dict()

    px2 = px.rename(columns={"product": "symbol"})
    fixed = []
    for d in DAYS:
        for sym in ["VELVETFRUIT_EXTRACT"]:
            tsub = tr[(tr["day"] == d) & (tr["symbol"] == sym)].sort_values("timestamp")
            psub = px2[(px2["day"] == d) & (px2["symbol"] == sym)].sort_values("timestamp")
            if tsub.empty or psub.empty:
                continue
            mm = pd.merge_asof(
                tsub,
                psub[["timestamp", "bid_price_1", "ask_price_1"]],
                on="timestamp",
                direction="backward",
            )
            mm["day"] = d
            mm["symbol"] = sym
            fixed.append(mm)
    mtr = pd.concat(fixed, ignore_index=True)
    mtr["joint_tight"] = [
        jmap.get((int(r["day"]), int(r["timestamp"])), False)
        for _, r in mtr.iterrows()
    ]
    mtr["buy_aggr"] = mtr["price"].astype(float) >= mtr["ask_price_1"].astype(float) - 1e-9
    sub = mtr[(mtr["buyer"] == "Mark 67") & mtr["buy_aggr"]]
    tight, loose = [], []
    for _, r in sub.iterrows():
        d, t0 = int(r["day"]), int(r["timestamp"])
        ts, mid = series_pair(px, d, "VELVETFRUIT_EXTRACT")
        dlt = forward_delta(ts, mid, t0, 5)
        if np.isnan(dlt):
            continue
        if bool(r["joint_tight"]):
            tight.append(dlt)
        else:
            loose.append(dlt)
    a, b = np.array(tight), np.array(loose)
    lines = [
        f"n_joint_tight={len(a)} mean_fwd5_extract={a.mean() if len(a) else float('nan'):.6g} frac_pos={(a>0).mean() if len(a) else float('nan'):.4f}",
        f"n_not_tight={len(b)} mean_fwd5_extract={b.mean() if len(b) else float('nan'):.6g} frac_pos={(b>0).mean() if len(b) else float('nan'):.4f}",
    ]
    (OUT / "r4_p3_mark67_extract_fwd5_joint_gate.txt").write_text("\n".join(lines) + "\n")

    # Spread-spread corr (inclineGod) on extract timestamps (no precomputed `spread` col in R4 tape)
    ex = px[px["product"] == "VELVETFRUIT_EXTRACT"][
        ["day", "timestamp", "ask_price_1", "bid_price_1"]
    ].copy()
    ex["s_ext"] = ex["ask_price_1"].astype(float) - ex["bid_price_1"].astype(float)
    ex = ex[["day", "timestamp", "s_ext"]].merge(sp52, on=["day", "timestamp"]).merge(sp53, on=["day", "timestamp"])

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 11 or np.std(a) <= 0 or np.std(b) <= 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    c52 = _corr(ex["s_ext"].to_numpy(), ex["s52"].to_numpy())
    c53 = _corr(ex["s_ext"].to_numpy(), ex["s53"].to_numpy())
    ccross = _corr(ex["s52"].to_numpy(), ex["s53"].to_numpy())
    (OUT / "r4_p3_spread_spread_corr.txt").write_text(
        f"corr(s_ext,s52)={c52}\ncorr(s_ext,s53)={c53}\ncorr(s52,s53)={ccross}\n"
    )

    print("phase3 ok")


if __name__ == "__main__":
    main()
