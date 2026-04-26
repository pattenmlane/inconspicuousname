"""
Leave-one-day-out stability: Mark 01 -> Mark 22 trades, Sonic joint gate, extract forwards.

For each holdout day d, aggregate mean fwd_5 / fwd_20 on remaining days only (pooled VEV rows,
tight=True). Compare to full-sample means to see if basket-level markouts are driven by one day.
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


def gate_frame(days: list[int]) -> pd.DataFrame:
    rows = []
    for day in days:
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


def extract_forwards(days: list[int]) -> pd.DataFrame:
    px = []
    for day in days:
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


def m01_m22_vev_rows(days: list[int]) -> pd.DataFrame:
    gate = gate_frame(days)
    xf = extract_forwards(days)
    trs = []
    for day in days:
        t = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        t["day"] = day
        trs.append(t)
    tr = pd.concat(trs, ignore_index=True)
    tr = tr.rename(columns={"symbol": "product"})
    b = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22")].copy()
    b = b.merge(gate, on=["day", "timestamp"], how="left")
    b = b.merge(xf, on=["day", "timestamp"], how="left")
    b["is_vev"] = b["product"].astype(str).str.startswith("VEV_")
    b = b[b["tight"] == True].copy()
    b = b[b["is_vev"]].copy()
    return b


def main() -> None:
    days = [1, 2, 3]
    full = m01_m22_vev_rows(days)
    rows = []
    rows.append(
        {
            "holdout": "none_full_sample",
            "n": len(full),
            "m5": float(full["fwd_5"].mean()),
            "m20": float(full["fwd_20"].mean()),
        }
    )
    for d in days:
        sub = full[full["day"] != d]
        rows.append(
            {
                "holdout": f"day_{d}",
                "n": len(sub),
                "m5": float(sub["fwd_5"].mean()) if len(sub) else float("nan"),
                "m20": float(sub["fwd_20"].mean()) if len(sub) else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out_path = OUT / "r4_m01_m22_vev_tight_loo_fwd_extract.csv"
    out.to_csv(out_path, index=False)
    # By-day means for reference
    byd = full.groupby("day").agg(n=("fwd_5", "count"), m5=("fwd_5", "mean"), m20=("fwd_20", "mean")).reset_index()
    byd.to_csv(OUT / "r4_m01_m22_vev_tight_by_day_fwd_extract.csv", index=False)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
