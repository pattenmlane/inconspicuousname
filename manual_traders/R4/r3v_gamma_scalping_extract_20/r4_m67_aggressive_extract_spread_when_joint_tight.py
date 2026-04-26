#!/usr/bin/env python3
"""
At timestamps with Sonic joint-tight 5200+5300, count Mark 67 **aggressive** extract
buys (price >= L1 ask) and the distribution of **VELVETFRUIT_EXTRACT** L1 spread.

Output: analysis_outputs/r4_m67_aggr_extract_spread_joint_tight_by_day.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = (1, 2, 3)
EXTRACT = "VELVETFRUIT_EXTRACT"
SURFACE = ("VEV_5200", "VEV_5300")
SPREAD_TH = 2
MARK67 = "Mark 67"


def main() -> None:
    rows = []
    for d in DAYS:
        px = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        sub = px[px["product"].isin(SURFACE + (EXTRACT,))].copy()
        sub["spr_vev"] = np.where(
            sub["product"].isin(SURFACE),
            pd.to_numeric(sub["ask_price_1"], errors="coerce")
            - pd.to_numeric(sub["bid_price_1"], errors="coerce"),
            np.nan,
        )
        pvt = sub[sub["product"].isin(SURFACE)].pivot_table(
            index="timestamp", columns="product", values="spr_vev", aggfunc="first"
        )
        pvt = pvt.dropna()
        jt = (pvt["VEV_5200"] <= SPREAD_TH) & (pvt["VEV_5300"] <= SPREAD_TH)
        joint_ts = set(jt[jt].index.astype(int))
        exr = sub[sub["product"] == EXTRACT][
            ["timestamp", "bid_price_1", "ask_price_1"]
        ].copy()
        exr["ts"] = exr["timestamp"].astype(int)
        exr["ask"] = pd.to_numeric(exr["ask_price_1"], errors="coerce")
        exr["spr"] = exr["ask"] - pd.to_numeric(exr["bid_price_1"], errors="coerce")
        g = exr.groupby("ts", as_index=True).first()
        ex_map = {int(t): (float(r["ask"]), float(r["spr"])) for t, r in g.iterrows()}

        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr = tr[(tr["symbol"] == EXTRACT) & (tr["buyer"].fillna("").astype(str) == MARK67)]
        n_hit = 0
        n_gt3 = 0
        n_gt4 = 0
        for _, r in tr.iterrows():
            t = int(r["timestamp"])
            if t not in joint_ts or t not in ex_map:
                continue
            pr = float(r["price"])
            ask, sp = ex_map[t]
            if pr < ask:
                continue
            n_hit += 1
            if sp > 3:
                n_gt3 += 1
            if sp > 4:
                n_gt4 += 1
        rows.append(
            {
                "day": d,
                "n_m67_aggr_joint_tight": n_hit,
                "n_extract_spread_gt3": n_gt3,
                "n_extract_spread_gt4": n_gt4,
                "share_gt3": round(n_gt3 / n_hit, 6) if n_hit else 0.0,
                "share_gt4": round(n_gt4 / n_hit, 6) if n_hit else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_m67_aggr_extract_spread_joint_tight_by_day.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
