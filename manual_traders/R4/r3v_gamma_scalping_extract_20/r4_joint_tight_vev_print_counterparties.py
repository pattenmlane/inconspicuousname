#!/usr/bin/env python3
"""
Joint-tight VEV prints — counterparty mix by tape day (Round 4).

For each trade on VEV_5200 or VEV_5300, require L1 spreads on **both** strikes
≤ SPREAD_TH at the same (day, timestamp) from prices_round_4_day_*.csv.
Then aggregate (buyer, seller) edges and top participants by day.

Outputs (analysis_outputs/):
  r4_joint_tight_vev_print_counts_by_day.csv
  r4_joint_tight_vev_print_top_edges_by_day.csv
  r4_joint_tight_vev_print_top_participants_by_day.csv
  r4_joint_tight_vev_print_run_log.txt
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "r4_joint_tight_vev_print_run_log.txt"

DAYS = (1, 2, 3)
SURFACE = ("VEV_5200", "VEV_5300")
SPREAD_TH = 2


def load_spreads() -> pd.DataFrame:
    rows = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        df = pd.read_csv(p, sep=";")
        if "day" not in df.columns:
            df["day"] = d
        sub = df[df["product"].isin(SURFACE)].copy()
        sub["bid_price_1"] = pd.to_numeric(sub["bid_price_1"], errors="coerce")
        sub["ask_price_1"] = pd.to_numeric(sub["ask_price_1"], errors="coerce")
        sub["spread"] = sub["ask_price_1"] - sub["bid_price_1"]
        rows.append(sub[["day", "timestamp", "product", "spread"]])
    return pd.concat(rows, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        df = pd.read_csv(p, sep=";")
        df["day"] = d
        frames.append(df)
    tr = pd.concat(frames, ignore_index=True)
    tr = tr[tr["symbol"].isin(SURFACE)].copy()
    tr["buyer"] = tr["buyer"].fillna("").astype(str)
    tr["seller"] = tr["seller"].fillna("").astype(str)
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    return tr


def main() -> None:
    spr = load_spreads()
    # pivot: one row per (day, ts) with both spreads
    pvt = spr.pivot_table(
        index=["day", "timestamp"],
        columns="product",
        values="spread",
        aggfunc="first",
    ).reset_index()
    pvt = pvt.dropna(subset=list(SURFACE))
    pvt["joint_tight"] = (pvt["VEV_5200"] <= SPREAD_TH) & (pvt["VEV_5300"] <= SPREAD_TH)
    tight_ts = pvt.loc[pvt["joint_tight"], ["day", "timestamp"]]

    tr = load_trades()
    m = tr.merge(tight_ts, on=["day", "timestamp"], how="inner")
    lines = []
    lines.append(f"Joint-tight (both spreads≤{SPREAD_TH}) VEV_5200/5300 prints: n={len(m)}")
    lines.append(f"By day: {m.groupby('day').size().to_dict()}")

    by_day = m.groupby("day").size().reset_index(name="n_prints")
    by_day.to_csv(OUT / "r4_joint_tight_vev_print_counts_by_day.csv", index=False)

    edge_rows = []
    for d in DAYS:
        sub = m[m["day"] == d]
        ctr = Counter(zip(sub["buyer"], sub["seller"]))
        for (bu, se), c in ctr.most_common(25):
            edge_rows.append({"day": d, "buyer": bu, "seller": se, "n": c})
    pd.DataFrame(edge_rows).to_csv(OUT / "r4_joint_tight_vev_print_top_edges_by_day.csv", index=False)

    part_rows = []
    for d in DAYS:
        sub = m[m["day"] == d]
        b = Counter(sub["buyer"])
        s = Counter(sub["seller"])
        for name, role in [("buyer", b), ("seller", s)]:
            for pty, c in role.most_common(15):
                part_rows.append({"day": d, "role": name, "participant": pty, "n": c})
    pd.DataFrame(part_rows).to_csv(OUT / "r4_joint_tight_vev_print_top_participants_by_day.csv", index=False)

    # Mark 01 share of prints (any side) by day
    m01 = []
    for d in DAYS:
        sub = m[m["day"] == d]
        if len(sub) == 0:
            m01.append({"day": d, "n": 0, "mark01_any_side": 0, "share": 0.0})
            continue
        mask = (sub["buyer"].str.contains("Mark 01", na=False)) | (
            sub["seller"].str.contains("Mark 01", na=False)
        )
        m01.append(
            {
                "day": d,
                "n": len(sub),
                "mark01_any_side": int(mask.sum()),
                "share": float(mask.mean()),
            }
        )
    pd.DataFrame(m01).to_csv(OUT / "r4_joint_tight_vev_print_mark01_share_by_day.csv", index=False)
    lines.append("Mark 01 on ≥1 side of print (5200/5300, joint-tight ts):")
    lines.append(pd.DataFrame(m01).to_string(index=False))

    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
