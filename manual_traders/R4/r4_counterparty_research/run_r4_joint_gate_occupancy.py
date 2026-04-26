#!/usr/bin/env python3
"""
Round 4 tape: fraction of price timestamps where Sonic joint gate is on
(VEV_5200 spread <= TH and VEV_5300 spread <= TH), inner-joining 5200+5300+extract rows.

Outputs JSON + CSV for day-stability context (pairs with Phase 3 forward-U splits).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
TH = 2
DAYS = (1, 2, 3)


def one_spread(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates("timestamp", keep="first")
        .sort_values("timestamp")
    )
    b = pd.to_numeric(v["bid_price_1"], errors="coerce")
    a = pd.to_numeric(v["ask_price_1"], errors="coerce")
    day_col = v["day"] if "day" in v.columns else int(df["day"].iloc[0])
    days = v["day"].astype(int) if "day" in v.columns else day_col
    return pd.DataFrame(
        {
            "day": days,
            "timestamp": v["timestamp"].astype(int),
            "spread": (a - b).astype(float),
        }
    )


def aligned(df: pd.DataFrame) -> pd.DataFrame:
    a = one_spread(df, "VEV_5200").rename(columns={"spread": "s5200"})
    b = one_spread(df, "VEV_5300").rename(columns={"spread": "s5300"})
    u = one_spread(df, "VELVETFRUIT_EXTRACT").rename(columns={"spread": "s_u"})
    m = a.merge(b, on=["day", "timestamp"], how="inner").merge(
        u[["day", "timestamp", "s_u"]], on=["day", "timestamp"], how="inner"
    )
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m


def main() -> None:
    rows = []
    all_parts = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        m = aligned(df)
        all_parts.append(m)
        rows.append(
            {
                "day": d,
                "n_aligned": int(len(m)),
                "n_tight": int(m["tight"].sum()),
                "P_tight": float(m["tight"].mean()),
                "median_s5200": float(m["s5200"].median()),
                "median_s5300": float(m["s5300"].median()),
                "median_s_u": float(m["s_u"].median()),
            }
        )
    pool = pd.concat(all_parts, ignore_index=True)
    pooled = {
        "n_aligned": int(len(pool)),
        "n_tight": int(pool["tight"].sum()),
        "P_tight": float(pool["tight"].mean()),
        "median_s5200": float(pool["s5200"].median()),
        "median_s5300": float(pool["s5300"].median()),
        "median_s_u": float(pool["s_u"].median()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    js = {"by_day": rows, "pooled": pooled, "TH": TH}
    (OUT / "r4_joint_gate_occupancy.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "r4_joint_gate_occupancy_by_day.csv", index=False)
    print("wrote", OUT / "r4_joint_gate_occupancy.json")


if __name__ == "__main__":
    main()
