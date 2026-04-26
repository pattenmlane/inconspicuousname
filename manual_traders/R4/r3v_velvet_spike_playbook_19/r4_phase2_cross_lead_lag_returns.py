#!/usr/bin/env python3
"""
Round 4 Phase 2 — Cross-instrument **lead–lag** on aligned price tape (mid returns).

Panel: inner-join all products on (day, timestamp) where VELVETFRUIT_EXTRACT has a row
(same convention as phase3 inner-join: one row per timestamp per product from prices CSV).

For each product P in {HYDROGEL_PACK, VEV_*} and each lag L in 0..MAX_LAG:
  rho(L) = Pearson corr( r_P.shift(L), r_EX )
where r_* = first difference of mid_price along time **within each day** (NaN at day boundaries).

Interpretation: **L > 0** — correlate **past** P return with **current** extract return ⇒ **P leads** extract by L rows.
**L = 0** — contemporaneous correlation.

Outputs:
- r4_phase2_leadlag_corr_extract_vs_products.csv — rows (scope, day, product, lag, corr, n)
  scope in {pooled, per_day}
- r4_phase2_leadlag_index.json — best lag per (product, scope) by |corr|
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
DAYS = (1, 2, 3)
EXTRACT = "VELVETFRUIT_EXTRACT"
MAX_LAG = 20
VEV_STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
OTHER = ("HYDROGEL_PACK",)


def load_prices(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    df["product"] = df["product"].astype(str)
    df["mid"] = pd.to_numeric(df["mid_price"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df[["day", "timestamp", "product", "mid"]].dropna(subset=["mid", "timestamp"])


def one_product_wide(day: int, product: str) -> pd.DataFrame:
    df = load_prices(day)
    v = (
        df[df["product"] == product]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")[["timestamp", "mid"]]
        .rename(columns={"mid": f"m_{product}"})
    )
    return v


def aligned_panel(day: int) -> pd.DataFrame:
    df = load_prices(day)
    ex = one_product_wide(day, EXTRACT)
    parts = [ex]
    for p in OTHER:
        parts.append(one_product_wide(day, p).rename(columns={f"m_{p}": f"m_{p}"}))
    for k in VEV_STRIKES:
        sym = f"VEV_{k}"
        parts.append(one_product_wide(day, sym))
    m = parts[0]
    for p in parts[1:]:
        m = m.merge(p, on="timestamp", how="inner")
    m["day"] = day
    return m.sort_values("timestamp").reset_index(drop=True)


def safe_corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    z = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(z)
    if n < 50:
        return float("nan"), n
    return float(z["a"].corr(z["b"])), n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    products = [EXTRACT, *OTHER, *[f"VEV_{k}" for k in VEV_STRIKES]]
    rows = []

    panels = {d: aligned_panel(d) for d in DAYS}

    def add_rows_for_frame(label_day: str, day_num: int | None, pan: pd.DataFrame) -> None:
        r_ex = pan[f"m_{EXTRACT}"].diff()
        for prod in products:
            if prod == EXTRACT:
                continue
            col = f"m_{prod}"
            if col not in pan.columns:
                continue
            r_p = pan[col].diff()
            for L in range(0, MAX_LAG + 1):
                lead = r_p.shift(L)
                c, n = safe_corr(lead, r_ex)
                rows.append(
                    {
                        "scope": label_day,
                        "day": day_num if day_num is not None else 0,
                        "product": prod,
                        "lag": L,
                        "corr": c,
                        "n": n,
                    }
                )

    for d in DAYS:
        add_rows_for_frame("per_day", d, panels[d])

    pooled = pd.concat([panels[d] for d in DAYS], ignore_index=True)
    add_rows_for_frame("pooled", None, pooled)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "r4_phase2_leadlag_corr_extract_vs_products.csv", index=False)

    # Best |corr| per product for pooled, lag>=1 (true lead)
    sub = out[(out["scope"] == "pooled") & (out["lag"] >= 1) & out["corr"].notna()].copy()
    sub["abs_c"] = sub["corr"].abs()
    best = sub.sort_values("abs_c", ascending=False).groupby("product").head(1)
    meta = {
        "max_lag": MAX_LAG,
        "definition": "r = mid.diff() within concatenated day panels; corr(r_other.shift(L), r_extract); L>0 => other leads extract.",
        "pooled_best_lag_ge1": best[["product", "lag", "corr", "n"]].to_dict(orient="records"),
        "note": "Contemporaneous (L=0) often dominates microstructure noise; L>=1 tests predictive lead.",
    }
    (OUT / "r4_phase2_leadlag_index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", len(out), "rows; pooled best L>=1:", json.dumps(meta["pooled_best_lag_ge1"][:8], indent=2))


if __name__ == "__main__":
    main()
