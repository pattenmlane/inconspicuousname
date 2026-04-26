#!/usr/bin/env python3
"""
Precompute timestamps: **VELVETFRUIT_EXTRACT** aggressive buy (price >= ask) with **Mark 49**
seller (Phase 1 Tier-A edge #2).

Output: precomputed/r4_extract_aggr_buy_m49_print.json

Run from repo root:
  python3 manual_traders/R4/r3v_jump_gap_filter_17/preprocess_r4_extract_aggr_buy_mark49.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "precomputed" / "r4_extract_aggr_buy_m49_print.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
EXTRACT = "VELVETFRUIT_EXTRACT"
SELLER = "Mark 49"
DAYS = (1, 2, 3)


def main() -> None:
    out: dict[str, list[int]] = {}
    for day in DAYS:
        pr = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        ex = pr[pr["product"] == EXTRACT].drop_duplicates("timestamp")
        bb = pd.to_numeric(ex["bid_price_1"], errors="coerce")
        ba = pd.to_numeric(ex["ask_price_1"], errors="coerce")
        bbo = ex.assign(bb=bb, ba=ba)[["timestamp", "bb", "ba"]]

        tr = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        tr = tr[(tr["symbol"] == EXTRACT) & (tr["seller"] == SELLER)].copy()
        tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
        m = tr.merge(bbo, on="timestamp", how="left")
        m = m.dropna(subset=["bb", "ba", "price"])
        aggr_buy = m["price"] >= m["ba"]
        out[str(day)] = sorted(m.loc[aggr_buy, "timestamp"].astype(int).unique().tolist())
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("Wrote", OUT, "counts:", {k: len(v) for k, v in out.items()}, "total", sum(len(v) for v in out.values()))


if __name__ == "__main__":
    main()
