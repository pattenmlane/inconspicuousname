"""
Tape: at each extract price timestamp, classify Sonic tier from 5200+5300 inner join:
  core = (s5200<=2)&(s5300<=2)
  wing_only = (s5200<=3)&(s5300<=3) & ~core
  loose = else

Mean extract mid fwd_20 (K=20 next rows, same as Phase 3) by (day, tier) and pooled.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
DAYS = [1, 2, 3]
EXTRACT = "VELVETFRUIT_EXTRACT"
V5200, V5300 = "VEV_5200", "VEV_5300"


def inner_spr() -> pd.DataFrame:
    rows = []
    for day in DAYS:
        ddf = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "bid_price_1", "ask_price_1"],
        )
        a = ddf[ddf["product"] == V5200].drop_duplicates("timestamp", keep="first")
        b = ddf[ddf["product"] == V5300].drop_duplicates("timestamp", keep="first")
        bid1 = pd.to_numeric(a["bid_price_1"], errors="coerce")
        ask1 = pd.to_numeric(a["ask_price_1"], errors="coerce")
        bid2 = pd.to_numeric(b["bid_price_1"], errors="coerce")
        ask2 = pd.to_numeric(b["ask_price_1"], errors="coerce")
        a = a.assign(
            s5200=(ask1 - bid1).astype(float),
            day=ddf["day"].iloc[0] if "day" in ddf.columns and len(ddf) else int(day),
        )
        b = b.assign(s5300=(ask2 - bid2).astype(float))
        if "day" not in a.columns:
            a["day"] = int(day)
        if "day" not in b.columns:
            b["day"] = int(day)
        j = a[["day", "timestamp", "s5200"]].merge(
            b[["day", "timestamp", "s5300"]], on=["day", "timestamp"], how="inner"
        )
        rows.append(j)
    return pd.concat(rows, ignore_index=True)


def extract_fwd20() -> pd.DataFrame:
    u_list = []
    for day in DAYS:
        ddf = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["day", "timestamp", "product", "mid_price"],
        )
        u = ddf[ddf["product"] == EXTRACT].drop_duplicates("timestamp", keep="first")
        u_list.append(u)
    u = pd.concat(u_list, ignore_index=True)
    u = u.sort_values(["day", "timestamp"])
    u["mid"] = pd.to_numeric(u["mid_price"], errors="coerce")
    u["fwd_20"] = u.groupby("day")["mid"].transform(lambda s: s.shift(-20) - s)
    return u[["day", "timestamp", "fwd_20"]]


def main() -> None:
    g = inner_spr()
    g["core"] = (g["s5200"] <= 2) & (g["s5300"] <= 2)
    g["wing_band"] = (g["s5200"] <= 3) & (g["s5300"] <= 3)
    g["tier"] = "loose"
    g.loc[g["core"], "tier"] = "core"
    g.loc[g["wing_band"] & ~g["core"], "tier"] = "wing_only"
    u = extract_fwd20().merge(g[["day", "timestamp", "tier"]], on=["day", "timestamp"], how="inner")
    u = u.dropna(subset=["fwd_20"])
    by = (
        u.groupby(["day", "tier"])
        .agg(n=("fwd_20", "count"), mean_fwd20=("fwd_20", "mean"), med_fwd20=("fwd_20", "median"))
        .reset_index()
    )
    by.to_csv(OUT / "r4_extract_fwd20_by_sonic_tier_by_day.csv", index=False)
    po = u.groupby("tier").agg(n=("fwd_20", "count"), mean_fwd20=("fwd_20", "mean")).reset_index()
    po.insert(0, "scope", "pooled")
    po.to_csv(OUT / "r4_extract_fwd20_by_sonic_tier_pooled.csv", index=False)
    print("wrote extract fwd20 by tier")


if __name__ == "__main__":
    main()
