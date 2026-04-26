"""
Per-day: at each extract price timestamp, joint Sonic tight flag; one-row-ahead extract mid change.

Outputs r4_extract_fwd1_mid_by_joint_day.csv — mean/std/count of (mid[t+1]-mid[t]) by (day, tight).
Tape-only; supports whether loose-book intervals have systematically different micro drift than tight.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
TH = 2
V5200, V5300 = "VEV_5200", "VEV_5300"
EXTRACT = "VELVETFRUIT_EXTRACT"


def gate_frame(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    rows = []
    for p in (V5200, V5300):
        v = df[df["product"] == p].drop_duplicates("timestamp", keep="first")
        bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": v["timestamp"].values,
                    "product": p,
                    "spr": (ask - bid).astype(float).values,
                }
            )
        )
    x = pd.concat(rows, ignore_index=True)
    a = x[x["product"] == V5200][["timestamp", "spr"]].rename(columns={"spr": "s5200"})
    b = x[x["product"] == V5300][["timestamp", "spr"]].rename(columns={"spr": "s5300"})
    m = a.merge(b, on="timestamp", how="inner")
    m["tight"] = (m["s5200"] <= TH) & (m["s5300"] <= TH)
    return m[["timestamp", "tight"]]


def main() -> None:
    agg_rows = []
    for day in (1, 2, 3):
        df = pd.read_csv(
            DATA / f"prices_round_4_day_{day}.csv",
            sep=";",
            usecols=["timestamp", "product", "mid_price"],
        )
        u = df[df["product"] == EXTRACT].drop_duplicates("timestamp", keep="first").sort_values("timestamp")
        u = u.assign(day=day)
        g = gate_frame(day)
        u = u.merge(g, on="timestamp", how="left")
        u["mid"] = pd.to_numeric(u["mid_price"], errors="coerce")
        u["fwd1"] = u["mid"].shift(-1) - u["mid"]
        u = u.dropna(subset=["tight", "fwd1"])
        for tight, sub in u.groupby("tight"):
            agg_rows.append(
                {
                    "day": day,
                    "tight": bool(tight),
                    "n": len(sub),
                    "mean_fwd1": float(sub["fwd1"].mean()),
                    "std_fwd1": float(sub["fwd1"].std(ddof=1)) if len(sub) > 1 else float("nan"),
                }
            )
    out = pd.DataFrame(agg_rows).sort_values(["day", "tight"], ascending=[True, False])
    path = OUT / "r4_extract_fwd1_mid_by_joint_day.csv"
    out.to_csv(path, index=False)
    print("wrote", path)


if __name__ == "__main__":
    main()
