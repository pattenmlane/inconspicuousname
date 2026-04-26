#!/usr/bin/env python3
"""Tape-only: Mark 22 as buyer vs seller on VEV_* prints, Round 4 days 1-3."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "r4_mark22_vev_roles_by_day.json"


def main() -> None:
    rows = []
    for day in (1, 2, 3):
        df = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        v = df[df["symbol"].astype(str).str.startswith("VEV_")].copy()
        v["m22_sell"] = (v["seller"].astype(str) == "Mark 22").astype(int)
        v["m22_buy"] = (v["buyer"].astype(str) == "Mark 22").astype(int)
        rows.append(
            {
                "day": day,
                "n_vev_prints": int(len(v)),
                "n_m22_seller": int(v["m22_sell"].sum()),
                "n_m22_buyer": int(v["m22_buy"].sum()),
                "frac_m22_seller": float(v["m22_sell"].mean()) if len(v) else 0.0,
                "frac_m22_buyer": float(v["m22_buy"].mean()) if len(v) else 0.0,
            }
        )
    pooled = {
        "n_vev_prints": sum(r["n_vev_prints"] for r in rows),
        "n_m22_seller": sum(r["n_m22_seller"] for r in rows),
        "n_m22_buyer": sum(r["n_m22_buyer"] for r in rows),
    }
    pooled["frac_m22_seller"] = pooled["n_m22_seller"] / max(pooled["n_vev_prints"], 1)
    pooled["frac_m22_buyer"] = pooled["n_m22_buyer"] / max(pooled["n_vev_prints"], 1)
    out = {"by_day": rows, "pooled": pooled}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
