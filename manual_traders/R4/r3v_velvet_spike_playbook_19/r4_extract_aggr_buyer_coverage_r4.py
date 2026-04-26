#!/usr/bin/env python3
"""Round 4: count aggressive **buys** on VELVETFRUIT_EXTRACT by buyer id (price >= ask1)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
DAYS = (1, 2, 3)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DAYS:
        pr = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        ex = pr[pr["product"] == "VELVETFRUIT_EXTRACT"].drop_duplicates("timestamp")
        bid = pd.to_numeric(ex["bid_price_1"], errors="coerce")
        ask = pd.to_numeric(ex["ask_price_1"], errors="coerce")
        book = pd.DataFrame({"timestamp": ex["timestamp"].values, "bid_ex": bid, "ask_ex": ask})
        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr = tr[tr["symbol"] == "VELVETFRUIT_EXTRACT"].copy()
        tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
        tr = tr.merge(book, on="timestamp", how="inner")
        ag = tr[tr["price"] >= tr["ask_ex"]]
        vc = ag["buyer"].astype(str).value_counts()
        for buyer, n in vc.items():
            rows.append({"day": d, "buyer": buyer, "aggr_buy_n": int(n)})
    out = pd.DataFrame(rows).sort_values(["day", "aggr_buy_n"], ascending=[True, False])
    out.to_csv(OUT / "r4_extract_aggr_buy_counts_by_buyer_day.csv", index=False)
    pivot = out.pivot_table(index="buyer", columns="day", values="aggr_buy_n", aggfunc="sum", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot.to_csv(OUT / "r4_extract_aggr_buy_counts_pivot.csv")
    (OUT / "r4_extract_aggr_buy_coverage_summary.json").write_text(
        json.dumps(
            {
                "definition": "Aggressive buy on VELVETFRUIT_EXTRACT: trade price >= ask1 same timestamp.",
                "top_buyers_total": pivot.head(15).to_dict(),
                "mark_01_aggr_buy_total": int(out.loc[out["buyer"] == "Mark 01", "aggr_buy_n"].sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pivot.head(12).to_string())
    print("Mark 01 aggressive buy count (all days):", int(out.loc[out["buyer"] == "Mark 01", "aggr_buy_n"].sum()))


if __name__ == "__main__":
    main()
