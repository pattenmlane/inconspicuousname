"""
Mark 01 -> Mark 22 trades × Sonic joint gate × forward extract mid (R4).

All trades with buyer Mark 01 and seller Mark 22 (any product). Merge gate at (day, ts).
Forward extract: K rows ahead in extract price tape per day.

Outputs:
  r4_m01_m22_fwd_extract_by_product_gate.csv
  r4_m01_m22_basket_size_same_ts.csv (how many distinct products per day,timestamp)
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
    a = x[x["product"] == V5200][["day", "timestamp", "spr"]].rename(columns={"spr": "s5200"})
    b = x[x["product"] == V5300][["day", "timestamp", "spr"]].rename(columns={"spr": "s5300"})
    m = a.merge(b, on=["day", "timestamp"], how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m[["day", "timestamp", "tight"]]


def extract_forwards() -> pd.DataFrame:
    px = []
    for day in DAYS:
        df = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "mid_price"],
        )
        px.append(df)
    u = pd.concat(px, ignore_index=True)
    u = u[u["product"] == EXTRACT].sort_values(["day", "timestamp"])
    for k in (5, 20):
        u[f"fwd_{k}"] = u.groupby("day")["mid_price"].transform(lambda s: s.astype(float).shift(-k) - s)
    return u[["day", "timestamp", "fwd_5", "fwd_20"]]


def main() -> None:
    gate = gate_frame()
    xf = extract_forwards()
    trs = []
    for day in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        t["day"] = day
        trs.append(t)
    tr = pd.concat(trs, ignore_index=True)
    tr = tr.rename(columns={"symbol": "product"})
    b = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22")].copy()
    b = b.merge(gate, on=["day", "timestamp"], how="left")
    b = b.merge(xf, on=["day", "timestamp"], how="left")
    b["is_vev"] = b["product"].astype(str).str.startswith("VEV_")

    g = (
        b.groupby(["product", "tight"])
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean"))
        .reset_index()
        .sort_values(["tight", "n"], ascending=[False, False])
    )
    g.to_csv(OUT / "r4_m01_m22_fwd_extract_by_product_gate.csv", index=False)

    agg = (
        b.groupby(["tight", "is_vev"])
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean"))
        .reset_index()
    )
    agg.to_csv(OUT / "r4_m01_m22_fwd_extract_vev_vs_other_gate.csv", index=False)

    bs = b.groupby(["day", "timestamp"])["product"].nunique().reset_index(name="n_syms")
    bs.to_csv(OUT / "r4_m01_m22_basket_size_same_ts.csv", index=False)
    bs2 = bs.groupby("n_syms").size().reset_index(name="count_ts")
    bs2.to_csv(OUT / "r4_m01_m22_basket_size_histogram.csv", index=False)

    print("wrote m01->m22 basket x gate tables to", OUT)


if __name__ == "__main__":
    main()
