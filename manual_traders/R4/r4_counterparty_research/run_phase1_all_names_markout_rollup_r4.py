#!/usr/bin/env python3
"""
Phase 1 — **every** distinct name appearing as buyer or seller: markout rollups (K=20).

- Rows: one per (name, role) where role is buyer_any | seller_any | aggr_buy | aggr_sell.
- Metrics: n, mean/median mark_20_sym, mark_20_u, frac positive; for aggr* only n>=5.

Input: r4_trades_enriched_markouts.csv
Output: r4_phase1_name_mark20_rollup.csv, r4_phase1_name_mark20_rollup.json (summary)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ENR = Path(__file__).resolve().parent / "outputs" / "r4_trades_enriched_markouts.csv"
OUT = Path(__file__).resolve().parent / "outputs"


def stats(s: pd.Series) -> dict:
    h = s.dropna()
    if len(h) == 0:
        return {"n": 0, "mean": None, "median": None, "frac_pos": None}
    h = h.astype(float)
    return {
        "n": int(len(h)),
        "mean": float(h.mean()),
        "median": float(h.median()),
        "frac_pos": float((h > 0).mean()),
    }


def main() -> None:
    df = pd.read_csv(ENR)
    for c in "mark_20_sym", "mark_20_u", "aggressor", "buyer", "seller":
        df[c] = df[c].astype(str) if c in ("aggressor", "buyer", "seller") else pd.to_numeric(df[c], errors="coerce")

    rows: list[dict] = []
    for role, gcol, mask in [
        ("buyer_any", "buyer", None),
        ("seller_any", "seller", None),
        ("aggr_buy", "buyer", df["aggressor"] == "buy"),
        ("aggr_sell", "seller", df["aggressor"] == "sell"),
    ]:
        sub = df if mask is None else df[mask]
        for name, g in sub.groupby(gcol):
            nm = str(name)
            if nm in ("nan", "None", ""):
                continue
            sym = g["mark_20_sym"]
            u = g["mark_20_u"]
            if role.startswith("aggr") and len(u.dropna()) < 5:
                continue
            d = {
                "role": role,
                "name": nm,
            }
            ss = stats(sym)
            su = stats(u)
            d["n_sym20"] = ss["n"]
            d["mean_sym20"] = ss["mean"]
            d["med_sym20"] = ss["median"]
            d["frac_pos_sym20"] = ss["frac_pos"]
            d["n_u20"] = su["n"]
            d["mean_u20"] = su["mean"]
            d["med_u20"] = su["median"]
            d["frac_pos_u20"] = su["frac_pos"]
            rows.append(d)

    t = pd.DataFrame(rows)
    t = t.sort_values(["role", "n_u20"], ascending=[True, False])
    t.to_csv(OUT / "r4_phase1_name_mark20_rollup.csv", index=False)
    n_names_buy = df["buyer"].nunique()
    n_names_sell = df["seller"].nunique()
    summ = {
        "distinct_buyers": int(n_names_buy),
        "distinct_sellers": int(n_names_sell),
        "rows_in_rollup": len(t),
        "note": "aggr_* rows omitted if n<5 for mark_20_u",
    }
    (OUT / "r4_phase1_name_mark20_rollup.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("wrote", OUT / "r4_phase1_name_mark20_rollup.csv", "rows", len(t))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
