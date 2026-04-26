"""
Follow-up: Mark 55 on VELVETFRUIT_EXTRACT × Sonic joint gate (R4 tape).

Reuses gate definition from analyze_r4_phase3_sonic_gate.py (inner join 5200+5300+extract
per day; tight = both leg spreads <= 2). Merges onto trades at (day, timestamp).

Outputs: r4_mark55_extract_fwd_by_gate.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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

    px_list: list[pd.DataFrame] = []
    for d in DAYS:
        p = pd.read_csv(
            DATA / f"prices_round_4_day_{d}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1", "mid_price"],
        )
        px_list.append(p)
    px = pd.concat(px_list, ignore_index=True)
    u = px[px["product"] == EXTRACT].copy()
    u = u.sort_values(["day", "timestamp"])
    for k in (5, 20):
        u[f"fwd_{k}"] = u.groupby("day")["mid_price"].transform(lambda s: s.shift(-k) - s)
    u = u.merge(gate, on=["day", "timestamp"], how="left")

    tr_list: list[pd.DataFrame] = []
    for d in DAYS:
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        tr_list.append(t)
    tr = pd.concat(tr_list, ignore_index=True)
    tr = tr.rename(columns={"symbol": "product"})
    tr = tr[tr["product"] == EXTRACT].copy()
    tr["price"] = tr["price"].astype(float)
    tr = tr.merge(
        px[(px["product"] == EXTRACT)][["day", "timestamp", "bid_price_1", "ask_price_1"]],
        on=["day", "timestamp"],
        how="left",
    )
    tr["aggr_buy"] = tr["price"] >= tr["ask_price_1"]
    tr["aggr_sell"] = tr["price"] <= tr["bid_price_1"]
    tr = tr.merge(gate, on=["day", "timestamp"], how="left")
    tr = tr.merge(u[["day", "timestamp", "fwd_5", "fwd_20"]], on=["day", "timestamp"], how="left")

    out: dict = {}
    for side, mask in [
        ("Mark55_aggr_buy", (tr["buyer"] == "Mark 55") & tr["aggr_buy"]),
        ("Mark55_aggr_sell", (tr["seller"] == "Mark 55") & tr["aggr_sell"]),
    ]:
        for gname, gm in [("tight", tr["tight"] == True), ("loose", tr["tight"] == False)]:
            sub = tr.loc[mask & gm]
            out[f"{side}_{gname}_fwd5"] = {
                "n": int(sub["fwd_5"].notna().sum()),
                "mean": float(sub["fwd_5"].mean()) if sub["fwd_5"].notna().any() else None,
            }
            out[f"{side}_{gname}_fwd20"] = {
                "n": int(sub["fwd_20"].notna().sum()),
                "mean": float(sub["fwd_20"].mean()) if sub["fwd_20"].notna().any() else None,
            }
    # by day Mark55 aggr buy tight
    byd = (
        tr.loc[(tr["buyer"] == "Mark 55") & tr["aggr_buy"] & (tr["tight"] == True)]
        .groupby("day")[["fwd_5", "fwd_20"]]
        .agg(["count", "mean"])
        .reset_index()
    )
    byd.to_csv(OUT / "r4_mark55_aggr_buy_tight_by_day.csv", index=False)
    out["Mark55_aggr_buy_tight_by_day_csv"] = "manual_traders/R4/r3v_velvet_beta_hedge_02/outputs/r4_mark55_aggr_buy_tight_by_day.csv"

    (OUT / "r4_mark55_extract_fwd_by_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT / "r4_mark55_extract_fwd_by_gate.json")


if __name__ == "__main__":
    main()
