"""
Tape: Mark 55 extract prints when Sonic joint is **loose** (trade-level merge).

For each extract trade, merge gate at (day,timestamp); require bid/ask for aggression.
Count rows with tight==False and Mark55 aggressive buy or sell; also share of all loose
extract prints (any counterparty).
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


def main() -> None:
    gate_frames = [gate_for_day(d) for d in DAYS]
    gate = pd.concat(gate_frames, ignore_index=True)

    px_list = []
    for d in DAYS:
        px_list.append(
            pd.read_csv(
                DATA / f"prices_round_4_day_{d}.csv",
                sep=";",
                usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1"],
            )
        )
    px = pd.concat(px_list, ignore_index=True)

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
    loose = tr[tr["tight"] == False].copy()
    loose["m55_touch"] = ((loose["buyer"] == "Mark 55") & loose["aggr_buy"]) | (
        (loose["seller"] == "Mark 55") & loose["aggr_sell"]
    )

    rows = []
    for day, g in loose.groupby("day"):
        n = len(g)
        n_touch = int(g["m55_touch"].sum())
        rows.append({"day": int(day), "n_extract_loose_trades": n, "n_m55_aggr_touch": n_touch, "share": n_touch / n if n else 0.0})

    summary = pd.DataFrame(rows)
    path = OUT / "r4_mark55_aggr_on_extract_loose_by_day.csv"
    summary.to_csv(path, index=False)
    pooled = pd.DataFrame(
        [
            {
                "day": "pooled",
                "n_extract_loose_trades": int(len(loose)),
                "n_m55_aggr_touch": int(loose["m55_touch"].sum()),
                "share": float(loose["m55_touch"].mean()) if len(loose) else 0.0,
            }
        ]
    )
    pooled.to_csv(OUT / "r4_mark55_aggr_on_extract_loose_pooled.csv", index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
