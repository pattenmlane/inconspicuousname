#!/usr/bin/env python3
"""
Round 3 price tapes: top-of-book spread ask1-bid1 for selected VEV strikes (microstructure for spread filter).
Output analysis_outputs/vev_spread_quantiles.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "vev_spread_quantiles.json"

TARGETS = ("VEV_5000", "VEV_5200", "VEV_5300")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def spread_row(r) -> float | None:
    b = r.get("bid_price_1")
    a = r.get("ask_price_1")
    try:
        if b is None or a is None or (isinstance(b, float) and np.isnan(b)):
            return None
        if isinstance(a, float) and np.isnan(a):
            return None
        return float(a) - float(b)
    except (TypeError, ValueError):
        return None


def main() -> None:
    out: dict = {"by_day_product": {}}
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        out["by_day_product"][str(day)] = {}
        for p in TARGETS:
            sub = df[df["product"] == p]
            sp = [s for _, row in sub.iterrows() for s in [spread_row(row)] if s is not None and s >= 0]
            if not sp:
                continue
            arr = np.asarray(sp, dtype=float)
            out["by_day_product"][str(day)][p] = {
                "n": int(len(arr)),
                "p10": float(np.quantile(arr, 0.1)),
                "p25": float(np.quantile(arr, 0.25)),
                "p50": float(np.quantile(arr, 0.5)),
                "p75": float(np.quantile(arr, 0.75)),
                "p90": float(np.quantile(arr, 0.9)),
                "frac_ge_3": float(np.mean(arr >= 3)),
                "frac_ge_4": float(np.mean(arr >= 4)),
                "frac_ge_5": float(np.mean(arr >= 5)),
                "frac_ge_6": float(np.mean(arr >= 6)),
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
