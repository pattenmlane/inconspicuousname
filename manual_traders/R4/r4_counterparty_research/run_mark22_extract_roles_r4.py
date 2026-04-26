#!/usr/bin/env python3
"""Tape: Mark 22 as buyer vs seller on VELVETFRUIT_EXTRACT prints, Round 4 days 1-3."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "r4_mark22_extract_roles_by_day.json"


def main() -> None:
    rows = []
    for day in (1, 2, 3):
        df = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        u = df[df["symbol"].astype(str) == "VELVETFRUIT_EXTRACT"].copy()
        u["m22_sell"] = (u["seller"].astype(str) == "Mark 22").astype(int)
        u["m22_buy"] = (u["buyer"].astype(str) == "Mark 22").astype(int)
        rows.append(
            {
                "day": day,
                "n_u_prints": int(len(u)),
                "n_m22_seller": int(u["m22_sell"].sum()),
                "n_m22_buyer": int(u["m22_buy"].sum()),
                "frac_m22_seller": float(u["m22_sell"].mean()) if len(u) else 0.0,
                "frac_m22_buyer": float(u["m22_buy"].mean()) if len(u) else 0.0,
            }
        )
    pooled = {
        "n_u_prints": sum(r["n_u_prints"] for r in rows),
        "n_m22_seller": sum(r["n_m22_seller"] for r in rows),
        "n_m22_buyer": sum(r["n_m22_buyer"] for r in rows),
    }
    pooled["frac_m22_seller"] = pooled["n_m22_seller"] / max(pooled["n_u_prints"], 1)
    pooled["frac_m22_buyer"] = pooled["n_m22_buyer"] / max(pooled["n_u_prints"], 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"by_day": rows, "pooled": pooled}, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT.with_suffix(".csv"), index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
