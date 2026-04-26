#!/usr/bin/env python3
"""Count aggressive buys (price>=ask1) with passive seller Mark22/Mark49 per VEV when Sonic joint gate is on."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [1, 2, 3]
SPREAD_TH = 2
G5200, G5300 = "VEV_5200", "VEV_5300"
PASSIVE = {"Mark 22", "Mark 49"}
SURFACE = ["VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]


def main() -> None:
    pr_parts = []
    tr_parts = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        t = DATA / f"trades_round_4_day_{d}.csv"
        if not p.is_file() or not t.is_file():
            continue
        pdf = pd.read_csv(p, sep=";")
        pdf["day"] = d
        pr_parts.append(pdf)
        tdf = pd.read_csv(t, sep=";")
        tdf["day"] = d
        tr_parts.append(tdf)
    pr = pd.concat(pr_parts, ignore_index=True)
    tr = pd.concat(tr_parts, ignore_index=True)

    pr = pr.rename(columns={"product": "symbol"})
    pr["bid1"] = pd.to_numeric(pr["bid_price_1"], errors="coerce")
    pr["ask1"] = pd.to_numeric(pr["ask_price_1"], errors="coerce")
    pr["spread"] = pr["ask1"] - pr["bid1"]

    gate = pr[pr["symbol"].isin([G5200, G5300])].copy()
    tight_keys = (
        gate.groupby(["day", "timestamp"])
        .apply(lambda g: (g["spread"] <= SPREAD_TH).all() and len(g) == 2, include_groups=False)
        .reset_index(name="tight")
    )
    tight_set = set(zip(tight_keys.loc[tight_keys["tight"], "day"], tight_keys.loc[tight_keys["tight"], "timestamp"]))

    m = tr.merge(pr[["day", "timestamp", "symbol", "bid1", "ask1"]], on=["day", "timestamp", "symbol"], how="inner")
    m["price"] = pd.to_numeric(m["price"], errors="coerce")
    m = m.dropna(subset=["price", "ask1", "seller"])
    m["aggr_buy"] = m["price"] >= m["ask1"]
    m["passive_hit"] = m["seller"].isin(PASSIVE)
    m["tight"] = m.apply(lambda r: (int(r["day"]), int(r["timestamp"])) in tight_set, axis=1)
    sub = m[m["tight"] & m["aggr_buy"] & m["passive_hit"]]

    rows = []
    for sym in SURFACE + ["VELVETFRUIT_EXTRACT"]:
        g = sub[sub["symbol"] == sym]
        rows.append({"symbol": sym, "n_prints": len(g)})
    out = pd.DataFrame(rows)
    path = OUT / "r4_v11_aggrbuy_m22_m49_counts_under_tight.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print("wrote", path)


if __name__ == "__main__":
    main()
