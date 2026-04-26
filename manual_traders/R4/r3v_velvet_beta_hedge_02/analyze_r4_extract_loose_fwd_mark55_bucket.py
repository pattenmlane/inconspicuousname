"""
Loose-book extract trades: forward mid change (K=5,20 rows in extract price tape) by
whether Mark 55 is aggressive on that print. Sonic joint gate at (day,timestamp).

Outputs:
  r4_extract_loose_fwd_by_m55_touch_by_day.csv
  r4_extract_loose_fwd_by_m55_touch_pooled.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
TH = 2
DAYS = [1, 2, 3]
EXTRACT = "VELVETFRUIT_EXTRACT"
V5200, V5300 = "VEV_5200", "VEV_5300"


def one_prod(df: pd.DataFrame, p: str) -> pd.DataFrame:
    v = df[df["product"] == p].drop_duplicates("timestamp", keep="first")
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    return pd.DataFrame(
        {"day": v["day"].values, "timestamp": v["timestamp"].values, "spr": (ask - bid).astype(float).values}
    )


def gate_for_day(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    a = one_prod(df, V5200).rename(columns={"spr": "s5200"})
    b = one_prod(df, V5300).rename(columns={"spr": "s5300"})
    m = a.merge(b, on=["day", "timestamp"], how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m[["day", "timestamp", "tight"]]


def extract_forwards() -> pd.DataFrame:
    px = []
    for d in DAYS:
        df = pd.read_csv(
            DATA / f"prices_round_4_day_{d}.csv",
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
    gate_frames = [gate_for_day(d) for d in DAYS]
    gate = pd.concat(gate_frames, ignore_index=True)
    xf = extract_forwards()

    px_u = []
    for d in DAYS:
        px_u.append(
            pd.read_csv(
                DATA / f"prices_round_4_day_{d}.csv",
                sep=";",
                usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1"],
            )
        )
    px = pd.concat(px_u, ignore_index=True)

    tr_list = []
    for d in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        tr_list.append(t)
    tr = pd.concat(tr_list, ignore_index=True).rename(columns={"symbol": "product"})
    tr = tr[tr["product"] == EXTRACT].copy()
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    tr = tr.merge(
        px[px["product"] == EXTRACT][["day", "timestamp", "bid_price_1", "ask_price_1"]],
        on=["day", "timestamp"],
        how="left",
    )
    tr["aggr_buy"] = tr["price"] >= pd.to_numeric(tr["ask_price_1"], errors="coerce")
    tr["aggr_sell"] = tr["price"] <= pd.to_numeric(tr["bid_price_1"], errors="coerce")
    tr = tr.merge(gate, on=["day", "timestamp"], how="left")
    tr = tr.merge(xf, on=["day", "timestamp"], how="left")
    loose = tr[tr["tight"] == False].copy()
    loose["m55_touch"] = ((loose["buyer"] == "Mark 55") & loose["aggr_buy"]) | (
        (loose["seller"] == "Mark 55") & loose["aggr_sell"]
    )

    by_day = (
        loose.groupby(["day", "m55_touch"], dropna=False)
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean"))
        .reset_index()
        .rename(columns={"m55_touch": "mark55_aggressive_touch"})
    )
    by_day["n"] = by_day["n"].astype(int)
    by_day.to_csv(OUT / "r4_extract_loose_fwd_by_m55_touch_by_day.csv", index=False)

    pooled = (
        loose.groupby("m55_touch", dropna=False)
        .agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean"))
        .reset_index()
        .rename(columns={"m55_touch": "mark55_aggressive_touch"})
    )
    pooled["n"] = pooled["n"].astype(int)
    pooled.insert(0, "scope", "pooled")
    pooled.to_csv(OUT / "r4_extract_loose_fwd_by_m55_touch_pooled.csv", index=False)
    print("wrote loose fwd buckets to", OUT)


if __name__ == "__main__":
    main()
