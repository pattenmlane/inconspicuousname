#!/usr/bin/env python3
"""
VEV surface 5100-5500: aggressive buy (price>=ask1) fwd+20 by passive seller,
under Sonic joint tight only. Complement to r4_v16 (aggr sell / passive buyer).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/workspace/Prosperity4Data/ROUND_4")
OUT = Path("/workspace/manual_traders/R4/r4_counterparty_phase1/outputs")
OUT.mkdir(parents=True, exist_ok=True)
TH = 2
DAYS = [1, 2, 3]
SURFACE = ["VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]


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
        for sym in SURFACE:
            h = (
                pr[(pr["day"] == d) & (pr["product"] == sym)]
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
            )
            if h.empty:
                continue
            ts = h["timestamp"].to_numpy(dtype=int)
            mid = pd.to_numeric(h["mid_price"], errors="coerce").to_numpy()
            bid1 = pd.to_numeric(h["bid_price_1"], errors="coerce").to_numpy()
            ask1 = pd.to_numeric(h["ask_price_1"], errors="coerce").to_numpy()
            imap = {int(ts[i]): i for i in range(len(ts))}

            tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
            tr["day"] = d
            tr = tr[tr["symbol"] == sym].merge(g, on=["day", "timestamp"], how="left")
            tr["tight"] = tr["tight"].fillna(False)
            tr = tr[tr["tight"]]

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
                if not buyer or price < aa:
                    continue
                fwd = float(mid[i + 20] - mid[i])
                rows.append({"day": d, "symbol": sym, "seller": seller, "fwd20": fwd})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_v17_vev_aggrbuy_tight_fwd20_rows.csv", index=False)
    summ = (
        df.groupby(["symbol", "seller"])["fwd20"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values(["mean", "count"], ascending=[False, False])
    )
    summ.to_csv(OUT / "r4_v17_vev_aggrbuy_tight_summary_by_symbol_seller.csv", index=False)
    print("Tight + aggressive buy (price>=ask1); seller = passive at ask")
    print(summ.head(40).to_string(index=False))
    print("...")
    print("lowest mean (worst for passive seller / adverse to lift):")
    print(summ.sort_values("mean").head(15).to_string(index=False))


if __name__ == "__main__":
    main()
