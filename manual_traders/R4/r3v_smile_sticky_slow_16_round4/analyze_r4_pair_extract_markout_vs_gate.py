"""
Round 4 — extract trades: K=20 forward mid markout by (buyer,seller), split Sonic tight vs wide.

Gate: same inner-join 5200/5300/extract as Phase 3 (TH=2). Only timestamps present in join get tight flag.

Output: analysis_outputs/r4_pair_extract_fwd20_vs_gate.json

Run: python3 manual_traders/R4/r3v_smile_sticky_slow_16_round4/analyze_r4_pair_extract_markout_vs_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "r4_pair_extract_fwd20_vs_gate.json"

TH = 2
K = 20
EX = "VELVETFRUIT_EXTRACT"
V520 = "VEV_5200"
V530 = "VEV_5300"

PAIRS = [
    ("Mark 14", "Mark 38"),
    ("Mark 38", "Mark 14"),
    ("Mark 01", "Mark 22"),
    ("Mark 55", "Mark 14"),
    ("Mark 14", "Mark 55"),
    ("Mark 01", "Mark 55"),
    ("Mark 55", "Mark 01"),
]


def days():
    return sorted(int(p.stem.split("_")[-1]) for p in DATA.glob("prices_round_4_day_*.csv"))


def one(df, prod):
    v = df[df["product"] == prod].drop_duplicates("timestamp").sort_values("timestamp")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    mid = pd.to_numeric(v["mid_price"], errors="coerce")
    return v.assign(bb=bid, ba=ask, m=mid)[["day", "timestamp", "bb", "ba", "m"]]


def gate_panel():
    parts = []
    for d in days():
        raw = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        raw["day"] = d
        a = one(raw, V520)[["day", "timestamp", "bb", "ba"]].rename(columns={"bb": "bb52", "ba": "ba52"})
        b = one(raw, V530)[["day", "timestamp", "bb", "ba"]].rename(columns={"bb": "bb53", "ba": "ba53"})
        e = one(raw, EX)[["day", "timestamp", "bb", "ba", "m"]].rename(columns={"bb": "bbu", "ba": "bau", "m": "m_ext"})
        m = a.merge(b, on=["day", "timestamp"], how="inner").merge(e, on=["day", "timestamp"], how="inner")
        s52 = (m["ba52"] - m["bb52"]).astype(float)
        s53 = (m["ba53"] - m["bb53"]).astype(float)
        m["tight"] = (s52 <= TH) & (s53 <= TH)
        parts.append(m[["day", "timestamp", "tight", "m_ext", "bbu", "bau"]])
    return pd.concat(parts, ignore_index=True)


def extract_mid_series():
    parts = []
    for d in days():
        raw = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        raw["day"] = d
        e = raw[raw["product"] == EX].drop_duplicates("timestamp").sort_values("timestamp")
        mid = pd.to_numeric(e["mid_price"], errors="coerce")
        parts.append(pd.DataFrame({"day": d, "timestamp": e["timestamp"].astype(int), "m": mid}))
    return pd.concat(parts, ignore_index=True)


def fwd20(ms: pd.DataFrame, day: int, ts: int) -> float | None:
    g = ms[(ms["day"] == day)].sort_values("timestamp")
    tss = g["timestamp"].to_numpy()
    mids = g["m"].astype(float).to_numpy()
    p = int(np.searchsorted(tss, ts, side="left"))
    if p >= len(tss) or tss[p] != ts:
        return None
    j = p + K
    if j >= len(tss):
        return None
    return float(mids[j] - mids[p])


def main():
    gate = gate_panel().set_index(["day", "timestamp"])
    ms = extract_mid_series()

    trs = []
    for d in days():
        p = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file():
            continue
        t = pd.read_csv(p, sep=";")
        t = t[t["symbol"] == EX].copy()
        t["day"] = int(d)
        trs.append(t)
    tr = pd.concat(trs, ignore_index=True)

    rows = []
    for _, r in tr.iterrows():
        d, ts = int(r["day"]), int(r["timestamp"])
        key = (d, ts)
        if key not in gate.index:
            continue
        rowg = gate.loc[key]
        tight = bool(rowg["tight"]) if isinstance(rowg, pd.Series) else bool(rowg.iloc[0]["tight"])
        bbu = float(rowg["bbu"]) if isinstance(rowg, pd.Series) else float(rowg.iloc[0]["bbu"])
        bau = float(rowg["bau"]) if isinstance(rowg, pd.Series) else float(rowg.iloc[0]["bau"])
        m0 = float(rowg["m_ext"]) if isinstance(rowg, pd.Series) else float(rowg.iloc[0]["m_ext"])
        pr = float(r["price"])
        buyer, seller = str(r["buyer"]), str(r["seller"])
        if pr >= bau:
            side = "buy_agg"
        elif pr <= bbu:
            side = "sell_agg"
        else:
            side = "inside"
        fv = fwd20(ms, d, ts)
        if fv is None:
            continue
        rows.append(
            {
                "day": d,
                "buyer": buyer,
                "seller": seller,
                "pair": f"{buyer}|{seller}",
                "tight": tight,
                "side": side,
                "fwd": fv,
            }
        )
    df = pd.DataFrame(rows)

    out_cells = []
    for buyer, seller in PAIRS:
        pkey = f"{buyer}|{seller}"
        for tight_lab, tval in [("tight", True), ("wide", False)]:
            for side in ("buy_agg", "sell_agg", "inside"):
                g = df[(df["buyer"] == buyer) & (df["seller"] == seller) & (df["tight"] == tval) & (df["side"] == side)]
                arr = g["fwd"].to_numpy()
                if len(arr) < 15:
                    continue
                out_cells.append(
                    {
                        "buyer": buyer,
                        "seller": seller,
                        "gate": tight_lab,
                        "side": side,
                        "n": int(len(arr)),
                        "mean_fwd20": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                    }
                )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"K": K, "cells": out_cells}, indent=2), encoding="utf-8")
    print("wrote", OUT, "n_rows", len(df))


if __name__ == "__main__":
    main()
