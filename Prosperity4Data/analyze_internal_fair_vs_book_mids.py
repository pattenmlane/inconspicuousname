#!/usr/bin/env python3
"""
Compare **internal fair** ``F(t) = E + PnL(t)`` (single-lot probe) to book-based
mids on the same timestamps.

Candidates (ASH_COATED_OSMIUM rows only):
  * ``micro_mid`` — best bid + best ask over L1–L3
  * ``wall_mid`` — min bid price + max ask price (winner-style)
  * ``popular_mid`` — max-volume bid + max-volume ask
  * ``jmerle_mid`` — max-volume bid + min-volume ask
  * ``csv_mid`` — ``mid_price`` column from the CSV

Metrics vs ``F``:
  * **RMSE** / **MAE** of level error ``(candidate - F)``
  * **corr** — Pearson correlation of levels (higher = same shape)
  * **ΔMAE** — mean abs error on **one-tick changes**: ``|Δc - ΔF|`` (lower = tracks moves)

Run from repo root::

  python3 Prosperity4Data/analyze_internal_fair_vs_book_mids.py \\
    --day-csv Prosperity4Data/ROUND1/prices_round_1_day_19.csv \\
    --pnl-log INK_INFO/248329.log \\
    --entry-from-log INK_INFO/248329.log
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCT_OSMIUM = "ASH_COATED_OSMIUM"


def _levels(row: pd.Series, side: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for i in range(1, 4):
        p = row.get(f"{side}_price_{i}")
        v = row.get(f"{side}_volume_{i}")
        if pd.isna(p) or pd.isna(v) or float(v) <= 0:
            continue
        out.append((float(p), float(v)))
    return out


def micro_mid_row(row: pd.Series) -> float:
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    return (max(p for p, _ in b) + min(p for p, _ in a)) / 2.0


def wall_mid_row(row: pd.Series) -> float:
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    return (min(p for p, _ in b) + max(p for p, _ in a)) / 2.0


def popular_mid_row(row: pd.Series) -> float:
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    pop_b = max(b, key=lambda t: t[1])[0]
    pop_a = max(a, key=lambda t: t[1])[0]
    return (pop_b + pop_a) / 2.0


def jmerle_mid_row(row: pd.Series) -> float:
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    pop_b = max(b, key=lambda t: t[1])[0]
    thin_a = min(a, key=lambda t: t[1])[0]
    return (pop_b + thin_a) / 2.0


def entry_price_from_log(log_path: Path, product: str = PRODUCT_OSMIUM) -> float:
    obj = json.loads(log_path.read_text(encoding="utf-8"))
    th = obj.get("tradeHistory")
    if not isinstance(th, list):
        raise SystemExit("log JSON missing tradeHistory list")
    for tr in th:
        if tr.get("buyer") != "SUBMISSION":
            continue
        if tr.get("symbol") != product:
            continue
        if int(tr.get("quantity", 0)) <= 0:
            continue
        return float(tr["price"])
    raise SystemExit(f"No SUBMISSION buy for {product} in {log_path}")


def load_pnl_from_log(log_path: Path) -> pd.Series:
    obj = json.loads(log_path.read_text(encoding="utf-8"))
    al = obj.get("activitiesLog")
    if not isinstance(al, str):
        raise SystemExit("log JSON missing activitiesLog string")
    dlog = pd.read_csv(io.StringIO(al), sep=";")
    return dlog.set_index(["timestamp", "product"])["profit_and_loss"]


def metrics(name: str, c: np.ndarray, f: np.ndarray) -> dict[str, float]:
    m = np.isfinite(c) & np.isfinite(f)
    if m.sum() < 5:
        return {"name": name, "n": float(m.sum())}
    ce, fe = c[m], f[m]
    err = ce - fe
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(ce, fe)[0, 1])
    dc = np.diff(ce, prepend=np.nan)
    dfv = np.diff(fe, prepend=np.nan)
    m2 = np.isfinite(dc) & np.isfinite(dfv)
    dmae = float(np.mean(np.abs(dc[m2] - dfv[m2]))) if m2.sum() > 2 else float("nan")
    return {
        "name": name,
        "n": float(m.sum()),
        "RMSE": rmse,
        "MAE": mae,
        "corr": corr,
        "delta_MAE": dmae,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-csv", type=Path, required=True)
    ap.add_argument("--entry-price", type=float, default=None)
    ap.add_argument("--entry-from-log", type=Path, default=None)
    ap.add_argument("--pnl-log", type=Path, default=None, help="Merge PnL from .log when CSV is zeros")
    args = ap.parse_args()

    if not args.day_csv.is_file():
        sys.exit(f"Missing {args.day_csv}")

    if args.entry_from_log:
        entry = entry_price_from_log(args.entry_from_log)
    elif args.entry_price is not None:
        entry = float(args.entry_price)
    else:
        sys.exit("Need --entry-price or --entry-from-log")

    df = pd.read_csv(args.day_csv, sep=";")
    pnl = pd.to_numeric(df["profit_and_loss"], errors="coerce")
    if args.pnl_log:
        if not args.pnl_log.is_file():
            sys.exit(f"Missing {args.pnl_log}")
        ser = load_pnl_from_log(args.pnl_log)

        def lk(r: pd.Series) -> float:
            k = (int(r["timestamp"]), str(r["product"]))
            return float(ser.loc[k]) if k in ser.index else float("nan")

        pnl = df.apply(lk, axis=1)

    osm = df.loc[df["product"] == PRODUCT_OSMIUM].sort_values("timestamp")
    osm = osm.assign(pnl=pnl.reindex(osm.index))
    osm = osm.reset_index(drop=True)
    osm["internal_fair"] = entry + osm["pnl"]
    osm["micro_mid"] = osm.apply(micro_mid_row, axis=1)
    osm["wall_mid"] = osm.apply(wall_mid_row, axis=1)
    osm["popular_mid"] = osm.apply(popular_mid_row, axis=1)
    osm["jmerle_mid"] = osm.apply(jmerle_mid_row, axis=1)
    osm["csv_mid"] = pd.to_numeric(osm["mid_price"], errors="coerce")

    F = osm["internal_fair"].to_numpy(dtype=float)
    cols = ["micro_mid", "wall_mid", "popular_mid", "jmerle_mid", "csv_mid"]
    mats = {c: osm[c].to_numpy(dtype=float) for c in cols}
    common = np.isfinite(F)
    for c in cols:
        common &= np.isfinite(mats[c])
    n_common = int(common.sum())

    rows = []
    for col in cols:
        rows.append(metrics(col, mats[col], F))
    tab = pd.DataFrame(rows).sort_values("RMSE")

    rows_c = []
    if n_common > 5:
        Fc, Cc = F[common], {c: mats[c][common] for c in cols}
        for col in cols:
            rows_c.append(metrics(col + " @common", Cc[col], Fc))
    tab_c = pd.DataFrame(rows_c).sort_values("RMSE") if rows_c else pd.DataFrame()

    print(f"Product: {PRODUCT_OSMIUM}  entry E={entry:g}")
    print(f"Internal fair F = E + PnL (PnL from {'--pnl-log' if args.pnl_log else 'CSV'})")
    print(f"Rows: {len(osm)} osmium ticks; all candidates + F finite on **{n_common}** rows.")
    print()
    print("--- Per candidate (finite F and finite that candidate) ---")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    if not tab_c.empty:
        print()
        print("--- Same rows where **every** candidate + F is finite (apples-to-apples) ---")
        print(tab_c.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print()
    print("Interpretation: lower RMSE/MAE = closer **levels** to F; higher corr = same **shape**;")
    print("lower delta_MAE = closer **tick-to-tick moves** to F.")


if __name__ == "__main__":
    main()
