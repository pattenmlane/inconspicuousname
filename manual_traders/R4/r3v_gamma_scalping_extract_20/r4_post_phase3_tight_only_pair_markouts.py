#!/usr/bin/env python3
"""
Post–Phase 3: counterparty pair markouts **restricted to Sonic joint-tight**
timestamps only (same inner-join + TH=2 as Phase 3).

Outputs:
  analysis_outputs/r4_tight_only_pair_markout_by_day.csv
  analysis_outputs/r4_tight_only_pair_markout_pooled.csv
"""
from __future__ import annotations

import bisect
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
TH = 2
KS = (5, 20, 100)
EXTRACT = "VELVETFRUIT_EXTRACT"


def load_px() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            if "day" not in df.columns:
                df["day"] = d
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_tr() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            t = pd.read_csv(p, sep=";")
            t["day"] = d
            frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    return tr


def one_prod(px: pd.DataFrame, day: int, prod: str) -> pd.DataFrame:
    v = px[(px["day"] == day) & (px["product"] == prod)].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return pd.DataFrame(
        {
            "day": day,
            "timestamp": v["timestamp"].astype(int),
            "spread": (ask - bid).astype(float),
            "mid": mid.astype(float),
        }
    )


def tight_panel(px: pd.DataFrame, day: int) -> pd.DataFrame:
    a = one_prod(px, day, "VEV_5200").rename(columns={"spread": "s5200"})
    b = one_prod(px, day, "VEV_5300").rename(columns={"spread": "s5300"})
    e = one_prod(px, day, EXTRACT)[["day", "timestamp"]]
    m = a.merge(b, on=["day", "timestamp"], how="inner").merge(e, on=["day", "timestamp"], how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m[["day", "timestamp", "tight"]]


def prep_mid(px: pd.DataFrame) -> tuple[dict[tuple[int, int, str], float], dict[int, np.ndarray]]:
    px["mid"] = pd.to_numeric(px["mid_price"], errors="coerce")
    lk: dict[tuple[int, int, str], float] = {}
    for _, r in px.iterrows():
        lk[(int(r["day"]), int(r["timestamp"]), str(r["product"]))] = float(r["mid"])
    ts = {}
    for d in DAYS:
        ts[d] = np.sort(px[px["day"] == d]["timestamp"].unique())
    return lk, ts


def fwd(lk: dict, tsu: np.ndarray, d: int, ts: int, sym: str, k: int) -> float | None:
    i = bisect.bisect_left(tsu, ts)
    if i >= len(tsu) or tsu[i] != ts:
        return None
    j = i + k
    if j >= len(tsu):
        return None
    t2 = int(tsu[j])
    a = lk.get((d, ts, sym))
    b = lk.get((d, t2, sym))
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return None
    return float(b - a)


def main() -> int:
    px = load_px()
    tr = load_tr()
    mid_lk, ts_sort = prep_mid(px)

    tight_ts: set[tuple[int, int]] = set()
    for d in DAYS:
        tp = tight_panel(px, d)
        for _, r in tp[tp["tight"]].iterrows():
            tight_ts.add((int(r["day"]), int(r["timestamp"])))

    pairs = [
        ("Mark 01", "Mark 22", "VEV_5200"),
        ("Mark 01", "Mark 22", "VEV_5300"),
        ("Mark 01", "Mark 22", "VEV_5400"),
        ("Mark 14", "Mark 55", EXTRACT),
        ("Mark 67", "Mark 22", EXTRACT),
    ]

    rows = []
    for buyer, seller, sym in pairs:
        sub = tr[(tr["buyer"] == buyer) & (tr["seller"] == seller) & (tr["symbol"] == sym)]
        for _, r in sub.iterrows():
            d, ts = int(r["day"]), int(r["timestamp"])
            if (d, ts) not in tight_ts:
                continue
            for K in KS:
                dm = fwd(mid_lk, ts_sort[d], d, ts, sym, K)
                if dm is None:
                    continue
                rows.append(
                    {
                        "buyer": buyer,
                        "seller": seller,
                        "symbol": sym,
                        "day": d,
                        "K": K,
                        "d_mid": dm,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        OUT.joinpath("r4_tight_only_pair_markout_pooled.csv").write_text("empty\n")
        return 0

    pooled = (
        df.groupby(["buyer", "seller", "symbol", "K"])["d_mid"]
        .agg(n="count", mean="mean", std="std", median="median")
        .reset_index()
    )
    pooled.to_csv(OUT / "r4_tight_only_pair_markout_pooled.csv", index=False)

    by_day = (
        df.groupby(["buyer", "seller", "symbol", "day", "K"])["d_mid"]
        .agg(n="count", mean="mean")
        .reset_index()
    )
    by_day.to_csv(OUT / "r4_tight_only_pair_markout_by_day.csv", index=False)
    print(pooled.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
