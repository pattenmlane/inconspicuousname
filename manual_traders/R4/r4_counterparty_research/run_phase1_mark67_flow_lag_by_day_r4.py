#!/usr/bin/env python3
"""Phase 1: Mark 67 signed aggressive extract flow vs future dU — correlation by day (lag 0-5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
M = "Mark 67"


def aggro(p: float, bid: float, ask: float) -> str:
    if p >= ask:
        return "buy"
    if p <= bid:
        return "sell"
    return "mid"


def main() -> None:
    pr_parts, tr_parts = [], []
    for d in (1, 2, 3):
        pr = pd.read_csv(DATA / f"prices_round_4_day_{d}.csv", sep=";")
        pr["day"] = d
        pr_parts.append(pr)
        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr["day"] = d
        tr_parts.append(tr)
    pr = pd.concat(pr_parts, ignore_index=True)
    tr = pd.concat(tr_parts, ignore_index=True)

    ubb = pr[pr["product"] == "VELVETFRUIT_EXTRACT"][
        ["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]
    ].copy()
    ubb["mid"] = pd.to_numeric(ubb["mid_price"], errors="coerce")
    ubb = ubb.sort_values(["day", "timestamp"])

    ut = tr[tr["symbol"] == "VELVETFRUIT_EXTRACT"].merge(
        ubb[["day", "timestamp", "bid_price_1", "ask_price_1"]],
        on=["day", "timestamp"],
        how="inner",
    )
    ut["ag"] = [
        aggro(float(p), float(bd), float(ak))
        for p, bd, ak in zip(
            ut["price"].astype(float),
            pd.to_numeric(ut["bid_price_1"], errors="coerce"),
            pd.to_numeric(ut["ask_price_1"], errors="coerce"),
        )
    ]
    ut["qty"] = ut["quantity"].astype(int)
    ut["f"] = np.where(
        (ut["ag"] == "buy") & (ut["buyer"] == M),
        ut["qty"],
        np.where((ut["ag"] == "sell") & (ut["seller"] == M), -ut["qty"], 0.0),
    )
    flow = ut.groupby(["day", "timestamp"], as_index=False)["f"].sum().rename(columns={"f": "flow"})

    rows = []
    for d in (1, 2, 3):
        ul = ubb[ubb["day"] == d].merge(flow[flow["day"] == d], on=["day", "timestamp"], how="left").fillna(
            {"flow": 0.0}
        )
        ul["d_mid"] = ul["mid"].diff()
        for L in range(0, 6):
            ul[f"du_L{L}"] = ul["d_mid"].shift(-L)
            sub = ul[["flow", f"du_L{L}"]].dropna()
            sub = sub[np.isfinite(sub["flow"]) & np.isfinite(sub[f"du_L{L}"])]
            if len(sub) > 50 and sub["flow"].std() > 1e-9 and sub[f"du_L{L}"].std() > 1e-9:
                rows.append(
                    {
                        "day": d,
                        "lag_ticks": L,
                        "corr": float(sub["flow"].corr(sub[f"du_L{L}"])),
                        "n": int(len(sub)),
                    }
                )

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "r4_phase1_mark67_signed_flow_lagcorr_by_day.csv", index=False)
    print("wrote", OUT / "r4_phase1_mark67_signed_flow_lagcorr_by_day.csv")


if __name__ == "__main__":
    main()
