"""
HYDROGEL_PACK trades × Sonic joint gate (R4): Mark 14↔38 duopoly forward mid.

Merge gate at (day, timestamp). Forward hydro mid = K rows ahead in price tape.
Pairs: (14,38) and (38,14) as buyer,seller.

Output: r4_hydro_pair_fwd_by_gate.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
DAYS = [1, 2, 3]
TH = 2
HYDRO = "HYDROGEL_PACK"
V5200, V5300 = "VEV_5200", "VEV_5300"


def gate_frame() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        for p in (V5200, V5300):
            v = df[df["product"] == p].drop_duplicates("timestamp", keep="first")
            bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
            ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
            rows.append(
                pd.DataFrame(
                    {
                        "day": day,
                        "timestamp": v["timestamp"].values,
                        "product": p,
                        "spr": (ask - bid).astype(float).values,
                    }
                )
            )
    x = pd.concat(rows, ignore_index=True)
    a = x[x["product"] == V5200][["day", "timestamp", "spr"]].rename(columns={"spr": "s5200"})
    b = x[x["product"] == V5300][["day", "timestamp", "spr"]].rename(columns={"spr": "s5300"})
    m = a.merge(b, on=["day", "timestamp"], how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m[["day", "timestamp", "tight"]]


def hydro_forwards() -> pd.DataFrame:
    px = []
    for day in DAYS:
        df = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "mid_price"],
        )
        px.append(df)
    h = pd.concat(px, ignore_index=True)
    h = h[h["product"] == HYDRO].sort_values(["day", "timestamp"])
    for k in (5, 20):
        h[f"fwd_{k}"] = h.groupby("day")["mid_price"].transform(lambda s: s.astype(float).shift(-k) - s)
    return h[["day", "timestamp", "fwd_5", "fwd_20"]]


def main() -> None:
    gate = gate_frame()
    hf = hydro_forwards()
    trs = []
    for day in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        t["day"] = day
        trs.append(t)
    tr = pd.concat(trs, ignore_index=True)
    tr = tr.rename(columns={"symbol": "product"})
    tr = tr[tr["product"] == HYDRO].copy()
    tr = tr.merge(gate, on=["day", "timestamp"], how="left")
    tr = tr.merge(hf, on=["day", "timestamp"], how="left")
    tr["pair"] = tr["buyer"].astype(str) + "->" + tr["seller"].astype(str)
    sub = tr[tr["pair"].isin(["Mark 14->Mark 38", "Mark 38->Mark 14"])]
    g = (
        sub.groupby(["pair", "tight"])
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean"))
        .reset_index()
    )
    g.to_csv(OUT / "r4_hydro_pair_fwd_by_gate.csv", index=False)
    byd = (
        sub.groupby(["day", "pair", "tight"])
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"))
        .reset_index()
    )
    byd.to_csv(OUT / "r4_hydro_pair_fwd_by_day_gate.csv", index=False)
    print("wrote", OUT / "r4_hydro_pair_fwd_by_gate.csv")


if __name__ == "__main__":
    main()
