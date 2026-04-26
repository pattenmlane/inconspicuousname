#!/usr/bin/env python3
"""HYDROGEL aggressive-sell fwd20 by passive buyer, split by Sonic joint gate (R4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/workspace/Prosperity4Data/ROUND_4")
OUT = Path("/workspace/manual_traders/R4/r4_counterparty_phase1/outputs")
TH = 2
DAYS = [1, 2, 3]


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def tight_panel(pr: pd.DataFrame, day: int) -> pd.DataFrame:
    p = pr[pr["day"] == day]
    a = (
        p[p["product"] == "VEV_5200"]
        .drop_duplicates("timestamp")
        .assign(
            s52=lambda x: pd.to_numeric(x["ask_price_1"], errors="coerce")
            - pd.to_numeric(x["bid_price_1"], errors="coerce")
        )[["timestamp", "s52"]]
    )
    b = (
        p[p["product"] == "VEV_5300"]
        .drop_duplicates("timestamp")
        .assign(
            s53=lambda x: pd.to_numeric(x["ask_price_1"], errors="coerce")
            - pd.to_numeric(x["bid_price_1"], errors="coerce")
        )[["timestamp", "s53"]]
    )
    m = a.merge(b, on="timestamp")
    m["tight"] = (m["s52"] <= TH) & (m["s53"] <= TH)
    return m[["timestamp", "tight"]].assign(day=day)


def main() -> None:
    pr = load_prices()
    rows = []
    for d in DAYS:
        g = tight_panel(pr, d)
        h = pr[(pr["day"] == d) & (pr["product"] == "HYDROGEL_PACK")].drop_duplicates("timestamp").sort_values("timestamp")
        ts = h["timestamp"].to_numpy(dtype=int)
        mid = pd.to_numeric(h["mid_price"], errors="coerce").to_numpy()
        bid1 = pd.to_numeric(h["bid_price_1"], errors="coerce").to_numpy()
        ask1 = pd.to_numeric(h["ask_price_1"], errors="coerce").to_numpy()
        imap = {int(ts[i]): i for i in range(len(ts))}

        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr["day"] = d
        tr = tr[tr["symbol"] == "HYDROGEL_PACK"].merge(g, on=["day", "timestamp"], how="left")
        tr["tight"] = tr["tight"].fillna(False)

        for _, r in tr.iterrows():
            t = int(r["timestamp"])
            if t not in imap:
                continue
            i = imap[t]
            if i + 20 >= len(mid):
                continue
            ba, aa = bid1[i], ask1[i]
            if np.isnan(ba) or np.isnan(aa) or aa <= ba:
                continue
            price = float(r["price"])
            buyer = str(r.get("buyer") or "")
            seller = str(r.get("seller") or "")
            if not seller or price > ba:
                continue
            fwd = float(mid[i + 20] - mid[i])
            rows.append({"day": d, "tight": bool(r["tight"]), "buyer": buyer, "fwd20": fwd})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_v9_hydro_aggrsell_fwd20_rows.csv", index=False)
    summ = df.groupby(["tight", "buyer"])["fwd20"].agg(["count", "mean"]).reset_index().sort_values(["tight", "mean"])
    summ.to_csv(OUT / "r4_v9_hydro_aggrsell_summary_by_tight_buyer.csv", index=False)
    print(summ.to_string())
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
