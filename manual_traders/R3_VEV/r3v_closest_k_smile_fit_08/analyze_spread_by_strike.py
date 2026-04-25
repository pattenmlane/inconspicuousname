#!/usr/bin/env python3
"""Top-of-book spread in ticks by product/strike; Round 3 tapes (price columns)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "spread_median_by_product.json"

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VEV = [f"VEV_{k}" for k in STRIKES]
OTHER = ("VELVETFRUIT_EXTRACT", "HYDROGEL_PACK")


def main() -> None:
    out: dict = {"method": "ask_price_1 - bid_price_1 per row; median over full tape per day."}
    for day in (0, 1, 2):
        p = REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(p, sep=";")
        sub = df[df["product"].isin(VEV + list(OTHER))]
        sub = sub.assign(
            spr=sub["ask_price_1"].astype(float) - sub["bid_price_1"].astype(float)
        )
        g = sub.groupby("product")["spr"].median()
        out[str(day)] = {k: float(g[k]) for k in VEV + list(OTHER) if k in g.index}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
