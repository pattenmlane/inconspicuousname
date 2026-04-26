#!/usr/bin/env python3
"""
Round 4 Phase 2 — Cross-instrument lead–lag using **aggressor-signed trade flow**
(not mid returns).

At each (day, timestamp, product): sum signed quantity
  +qty if aggressive buy (price >= ask1)
  -qty if aggressive sell (price <= bid1)
  0 if no aggressive print at that tick (or unknown).

Align to **VELVETFRUIT_EXTRACT** price timestamps (same day): r_ex = mid.diff()
along extract time order. For each other product P:
  rho(L) = corr( F_P.shift(L), r_ex )
where F_P is signed flow at the extract timestamp row (0 if absent).

L>0 => past signed aggressive flow in P predicts current extract return.

Outputs:
- r4_phase2_leadlag_signedflow_vs_extract.csv
- r4_phase2_leadlag_signedflow_index.json
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
PRODUCTS = [EXTRACT, *OTHER, *[f"VEV_{k}" for k in VEV_STRIKES]]


def load_book(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    df["product"] = df["product"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["bid1"] = pd.to_numeric(df["bid_price_1"], errors="coerce")
    df["ask1"] = pd.to_numeric(df["ask_price_1"], errors="coerce")
    df["mid"] = pd.to_numeric(df["mid_price"], errors="coerce")
    return df[["day", "timestamp", "product", "bid1", "ask1", "mid"]].dropna(subset=["timestamp"])


def load_trades(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
    df["day"] = day
    df["product"] = df["symbol"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    return df


def signed_flow_table(day: int) -> pd.DataFrame:
    book = load_book(day)
    tr = load_trades(day)
    m = tr.merge(
        book.rename(columns={"bid1": "bid1_px", "ask1": "ask1_px"}),
        on=["day", "timestamp", "product"],
        how="left",
    )
    buy_a = m["price"] >= m["ask1_px"]
    sell_a = m["price"] <= m["bid1_px"]
    m["signed"] = np.where(buy_a, m["quantity"], np.where(sell_a, -m["quantity"], np.nan))
    m = m.dropna(subset=["signed"])
    if m.empty:
        return pd.DataFrame(columns=["day", "timestamp", "product", "signed"])
    g = m.groupby(["day", "timestamp", "product"], as_index=False)["signed"].sum()
    return g


def extract_series(day: int) -> pd.DataFrame:
    b = load_book(day)
    ex = (
        b[b["product"] == EXTRACT]
        .drop_duplicates(subset=["timestamp"], keep="first")
        .sort_values("timestamp")[["day", "timestamp", "mid"]]
        .reset_index(drop=True)
    )
    ex["r_ex"] = ex["mid"].diff()
    return ex


def safe_corr(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    z = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(z)
    if n < 50:
        return float("nan"), n
    return float(z["a"].corr(z["b"])), n


def panel_for_day(day: int) -> pd.DataFrame:
    ex = extract_series(day)
    flow = signed_flow_table(day)
    out = ex[["day", "timestamp", "r_ex"]].copy()
    for p in PRODUCTS:
        if p == EXTRACT:
            continue
        sub = flow[flow["product"] == p][["timestamp", "signed"]].rename(columns={"signed": f"f_{p}"})
        out = out.merge(sub, on="timestamp", how="left")
        out[f"f_{p}"] = out[f"f_{p}"].fillna(0.0)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = {d: panel_for_day(d) for d in DAYS}
    pooled = pd.concat([panels[d] for d in DAYS], ignore_index=True)

    rows = []
    prods = [p for p in PRODUCTS if p != EXTRACT]

    def run_frame(label: str, day_num: int | None, pan: pd.DataFrame) -> None:
        r_ex = pan["r_ex"]
        for prod in prods:
            col = f"f_{prod}"
            if col not in pan.columns:
                continue
            fser = pan[col]
            for L in range(0, MAX_LAG + 1):
                c, n = safe_corr(fser.shift(L), r_ex)
                rows.append(
                    {
                        "scope": label,
                        "day": int(day_num) if day_num is not None else 0,
                        "product": prod,
                        "lag": L,
                        "corr": c,
                        "n": n,
                    }
                )

    for d in DAYS:
        run_frame("per_day", d, panels[d])
    run_frame("pooled", None, pooled)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r4_phase2_leadlag_signedflow_vs_extract.csv", index=False)

    sub = df[(df["scope"] == "pooled") & (df["lag"] >= 1) & df["corr"].notna()].copy()
    sub["abs_c"] = sub["corr"].abs()
    best = sub.sort_values("abs_c", ascending=False).groupby("product").head(1)
    meta = {
        "max_lag": MAX_LAG,
        "definition": "F_P = sum aggressive signed qty at (day,timestamp) on P; r_ex = extract mid.diff() on extract timestamps; corr(F_P.shift(L), r_ex).",
        "pooled_best_lag_ge1": best[["product", "lag", "corr", "n"]].to_dict(orient="records"),
    }
    (OUT / "r4_phase2_leadlag_signedflow_index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["pooled_best_lag_ge1"][:12], indent=2))


if __name__ == "__main__":
    main()
