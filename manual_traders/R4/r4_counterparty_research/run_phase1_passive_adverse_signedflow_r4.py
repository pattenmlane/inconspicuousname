#!/usr/bin/env python3
"""
Phase 1 supplement — bullet 5 (passive adverse selection proxy) + bullet 3 (signed flow lead-lag per Mark).

Passive side (tape):
- Aggressive **buy** (price >= ask): passive **seller** took the hit -> analyze markouts for rows where seller==Mark.
- Aggressive **sell** (price <= bid): passive **buyer** took the hit -> analyze markouts for rows where buyer==Mark.

Dominant pairs (from r4_graph_top_edges): Mark01->Mark22, Mark14<->Mark38, Mark55<->Mark01.
For each pair (B,S), split prints into passive-S (aggressor buy) and passive-B (aggressor sell), report
mean/median/n for mark_20_u and mark_20_sym (min n=15).

Signed aggressive extract flow per Mark:
- On VELVETFRUIT_EXTRACT trades only, at each (day, timestamp): for Mark M,
  flow_M = sum( qty if (aggressor==buy and buyer==M) else (-qty if aggressor==sell and seller==M) else 0 ).
- Merge onto extract price timeline; correlate flow_M with future dU at lags L=0..15 (same as Phase 2).

Outputs:
  r4_phase1_passive_markout_by_pair.csv
  r4_phase1_signed_extract_flow_per_mark_lagcorr.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
ENR = Path(__file__).resolve().parent / "outputs" / "r4_trades_enriched_markouts.csv"
OUT = Path(__file__).resolve().parent / "outputs"
MIN_N = 15
PAIRS = [
    ("Mark 01", "Mark 22"),
    ("Mark 14", "Mark 38"),
    ("Mark 38", "Mark 14"),
    ("Mark 55", "Mark 01"),
    ("Mark 01", "Mark 55"),
]
MARKS = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 55", "Mark 67", "Mark 49"]
DAYS = (1, 2, 3)


def aggro(p: float, bid: float, ask: float) -> str:
    if p >= ask:
        return "buy"
    if p <= bid:
        return "sell"
    return "mid"


def main() -> None:
    enr = pd.read_csv(ENR)
    for c in ("buyer", "seller", "aggressor", "symbol"):
        enr[c] = enr[c].astype(str)
    for c in ("mark_20_u", "mark_20_sym"):
        enr[c] = pd.to_numeric(enr[c], errors="coerce")

    passive_rows = []
    for b, s in PAIRS:
        sub = enr[(enr["buyer"] == b) & (enr["seller"] == s)]
        ps = sub[sub["aggressor"] == "buy"]  # passive seller s
        pb = sub[sub["aggressor"] == "sell"]  # passive buyer b
        for lab, g in [("passive_seller_on_aggr_buy", ps), ("passive_buyer_on_aggr_sell", pb)]:
            u = g["mark_20_u"].dropna()
            sy = g["mark_20_sym"].dropna()
            if len(u) < MIN_N:
                continue
            passive_rows.append(
                {
                    "buyer": b,
                    "seller": s,
                    "slice": lab,
                    "n": int(len(g)),
                    "n_u20": int(len(u)),
                    "mean_u20": float(u.mean()),
                    "median_u20": float(u.median()),
                    "frac_pos_u20": float((u > 0).mean()),
                    "mean_sym20": float(sy.mean()) if len(sy) >= MIN_N else float("nan"),
                    "median_sym20": float(sy.median()) if len(sy) >= MIN_N else float("nan"),
                }
            )
    pd.DataFrame(passive_rows).to_csv(OUT / "r4_phase1_passive_markout_by_pair.csv", index=False)

    # --- per-Mark signed flow on extract + lag corr ---
    pr_parts = []
    tr_parts = []
    for d in DAYS:
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
        suffixes=("", "_pr"),
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

    lag_rows = []
    for M in MARKS:
        ut[f"f_{M}"] = np.where(
            (ut["ag"] == "buy") & (ut["buyer"] == M),
            ut["qty"],
            np.where((ut["ag"] == "sell") & (ut["seller"] == M), -ut["qty"], 0.0),
        )
        flow = ut.groupby(["day", "timestamp"], as_index=False)[f"f_{M}"].sum().rename(columns={f"f_{M}": "flow"})
        uline = ubb.merge(flow, on=["day", "timestamp"], how="left").fillna({"flow": 0.0})
        uline["d_mid"] = uline.groupby("day")["mid"].diff()
        for L in range(0, 16):
            uline[f"du_L{L}"] = uline.groupby("day")["d_mid"].shift(-L)
            sub = uline[["flow", f"du_L{L}"]].dropna()
            sub = sub[np.isfinite(sub["flow"]) & np.isfinite(sub[f"du_L{L}"])]
            if len(sub) > 200 and sub["flow"].std() > 1e-9 and sub[f"du_L{L}"].std() > 1e-9:
                lag_rows.append(
                    {
                        "mark": M,
                        "lag_ticks": L,
                        "corr_flow_future_dU": float(sub["flow"].corr(sub[f"du_L{L}"])),
                        "n": int(len(sub)),
                    }
                )
        ut = ut.drop(columns=[f"f_{M}"], errors="ignore")

    pd.DataFrame(lag_rows).to_csv(OUT / "r4_phase1_signed_extract_flow_per_mark_lagcorr.csv", index=False)
    summ = {
        "pairs_analyzed": [list(p) for p in PAIRS],
        "passive_csv": str(OUT / "r4_phase1_passive_markout_by_pair.csv"),
        "lagcorr_csv": str(OUT / "r4_phase1_signed_extract_flow_per_mark_lagcorr.csv"),
    }
    (OUT / "r4_phase1_passive_signedflow_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("wrote passive + lagcorr outputs")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
