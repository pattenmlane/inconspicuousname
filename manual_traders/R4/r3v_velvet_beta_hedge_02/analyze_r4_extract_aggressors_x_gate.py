"""
Round 4 — all counterparty aggressors on VELVETFRUIT_EXTRACT × Sonic joint gate.

Aggressive buy: trade price >= ask1 at (day, ts). Participant = buyer.
Aggressive sell: trade price <= bid1. Participant = seller.

Merge gate (5200+5300 inner join, tight = both spreads <=2) at (day, timestamp).
Forward extract mid: K rows ahead in price tape (same as Phase 1).

Output: r4_extract_aggressor_fwd_by_gate.csv (pooled + filter n>=10)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
DAYS = [1, 2, 3]
TH = 2
V5200, V5300 = "VEV_5200", "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"


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
    p5200 = x[x["product"] == V5200][["day", "timestamp", "spr"]].rename(columns={"spr": "s5200"})
    p5300 = x[x["product"] == V5300][["day", "timestamp", "spr"]].rename(columns={"spr": "s5300"})
    m = p5200.merge(p5300, on=["day", "timestamp"], how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m[["day", "timestamp", "tight"]]


def extract_forwards() -> pd.DataFrame:
    px = []
    for day in DAYS:
        df = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1", "mid_price"],
        )
        px.append(df)
    u = pd.concat(px, ignore_index=True)
    u = u[u["product"] == EXTRACT].sort_values(["day", "timestamp"])
    for k in (5, 20):
        u[f"fwd_{k}"] = u.groupby("day")["mid_price"].transform(lambda s: s.astype(float).shift(-k) - s)
    return u[["day", "timestamp", "fwd_5", "fwd_20"]]


def main() -> None:
    gate = gate_frame()
    u = extract_forwards()
    trs = []
    for day in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        t["day"] = day
        trs.append(t)
    tr = pd.concat(trs, ignore_index=True)
    tr = tr.rename(columns={"symbol": "product"})
    tr = tr[tr["product"] == EXTRACT].copy()
    tr["price"] = tr["price"].astype(float)
    # bid/ask for aggression
    px_full = []
    for day in DAYS:
        px_full.append(
            pd.read_csv(
                DATA / f"prices_round_4_day_{day}.csv",
                sep=";",
                usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1"],
            )
        )
    px = pd.concat(px_full, ignore_index=True)
    ex = px[px["product"] == EXTRACT]
    tr = tr.merge(ex, on=["day", "timestamp", "product"], how="left")
    tr["aggr_buy"] = tr["price"] >= tr["ask_price_1"]
    tr["aggr_sell"] = tr["price"] <= tr["bid_price_1"]
    tr = tr.merge(gate, on=["day", "timestamp"], how="left")
    tr = tr.merge(u, on=["day", "timestamp"], how="left")
    rows = []
    for _, r in tr.iterrows():
        if r["aggr_buy"]:
            rows.append(
                {
                    "participant": r["buyer"],
                    "role": "aggr_buy",
                    "tight": bool(r["tight"]),
                    "fwd_5": r["fwd_5"],
                    "fwd_20": r["fwd_20"],
                    "day": r["day"],
                }
            )
        if r["aggr_sell"]:
            rows.append(
                {
                    "participant": r["seller"],
                    "role": "aggr_sell",
                    "tight": bool(r["tight"]),
                    "fwd_5": r["fwd_5"],
                    "fwd_20": r["fwd_20"],
                    "day": r["day"],
                }
            )
    ev = pd.DataFrame(rows)
    g = (
        ev.groupby(["participant", "role", "tight"])
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean"))
        .reset_index()
    )
    g.to_csv(OUT / "r4_extract_aggressor_fwd_by_gate.csv", index=False)
    g[g["n"] >= 15].sort_values("n", ascending=False).to_csv(
        OUT / "r4_extract_aggressor_fwd_by_gate_n15plus.csv", index=False
    )
    print("wrote", OUT / "r4_extract_aggressor_fwd_by_gate.csv")


if __name__ == "__main__":
    main()
