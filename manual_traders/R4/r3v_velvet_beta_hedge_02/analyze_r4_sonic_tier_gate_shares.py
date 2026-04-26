"""
Tape: inner-join 5200+5300 on (day,timestamp). Classify each row:
  core = (s5200<=2)&(s5300<=2)
  wing_only = (s5200<=3)&(s5300<=3) & ~core
  loose = else

Outputs r4_sonic_tier_gate_shares_by_day.csv and pooled row.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
DAYS = [1, 2, 3]
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


def main() -> None:
    m = inner_spr()
    m["core"] = (m["s5200"] <= 2) & (m["s5300"] <= 2)
    m["wing_band"] = (m["s5200"] <= 3) & (m["s5300"] <= 3)
    m["wing_only"] = m["wing_band"] & ~m["core"]
    m["loose"] = ~m["wing_band"]
    by = m.groupby("day").agg(
        n=("timestamp", "count"),
        p_core=("core", "mean"),
        p_wing_only=("wing_only", "mean"),
        p_loose=("loose", "mean"),
    ).reset_index()
    by.to_csv(OUT / "r4_sonic_tier_gate_shares_by_day.csv", index=False)
    pooled = pd.DataFrame(
        [
            {
                "day": "pooled",
                "n": len(m),
                "p_core": float(m["core"].mean()),
                "p_wing_only": float(m["wing_only"].mean()),
                "p_loose": float(m["loose"].mean()),
            }
        ]
    )
    pooled.to_csv(OUT / "r4_sonic_tier_gate_shares_pooled.csv", index=False)
    print("wrote tier gate shares")


if __name__ == "__main__":
    main()
