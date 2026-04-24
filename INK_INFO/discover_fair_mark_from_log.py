#!/usr/bin/env python3
"""
Infer what Prosperity's **mark-to-model** tracks, given a **website export** `.log`
(JSON with `activitiesLog` = semicolon CSV).

Method (needs a **known inventory** leg — e.g. buy **1** lot once, then hold):

1. Parse **ASH_COATED_OSMIUM** rows; read `profit_and_loss` and L1–L3 book.
2. Build **candidate marks** from the same book the strategy sees:
   **micro** (best touch mid), **wall** (min bid + max ask over positive levels),
   **popular** (max-volume bid + max-volume ask), **jmerle** (popular bid + thinnest ask).
3. Compare **ΔPnL** to **Δcandidate** (tick alignment). If marking were *exactly* that
   candidate with coefficient 1, corr would be ~1.
4. Fit **(PnL_t − PnL_0) ≈ a + b·(candidate_t − candidate_0)** — for a +1 lot, you
   want **b ≈ 1** and small RMSE if that series *is* the mark.
5. **Smooth** the candidate (rolling mean of **wall**): internal fair may be
   low-pass filtered; compare corr(ΔPnL, Δsmooth).
6. **Shape check**: cumsum(ΔPnL) vs candidate **levels** (detrended correlation) —
   picks up the right series even when level offsets/fees break simple OLS.

**VWAP** cannot be tested from this CSV alone (no per-tick trade tape); use
market-trades / custom logging if Prosperity exposes it elsewhere.

Usage:
  python3 INK_INFO/discover_fair_mark_from_log.py INK_INFO/248329.log
  python3 INK_INFO/discover_fair_mark_from_log.py path/to/export.log --product ASH_COATED_OSMIUM
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def detrended_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = np.arange(len(x))
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    x, y = x[mask], y[mask]
    n = np.arange(len(x))
    x0 = x - np.polyval(np.polyfit(n, x, 1), n)
    y0 = y - np.polyval(np.polyfit(n, y, 1), n)
    if np.std(x0) < 1e-12 or np.std(y0) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x0, y0)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare PnL to book-based fair candidates.")
    ap.add_argument("log_json", type=Path, help="Prosperity export .log (JSON)")
    ap.add_argument("--product", default="ASH_COATED_OSMIUM")
    args = ap.parse_args()

    obj = json.loads(args.log_json.read_text(encoding="utf-8"))
    al = obj.get("activitiesLog")
    if not isinstance(al, str):
        sys.exit("Missing activitiesLog string")

    df = pd.read_csv(io.StringIO(al), sep=";")
    if "product" not in df.columns:
        sys.exit("Unexpected CSV shape")

    s = df.loc[df["product"] == args.product].sort_values("timestamp").reset_index(drop=True)
    if len(s) < 20:
        sys.exit(f"Too few rows for {args.product!r}: {len(s)}")

    s["micro"] = s.apply(micro_mid_row, axis=1)
    s["wall"] = s.apply(wall_mid_row, axis=1)
    s["popular"] = s.apply(popular_mid_row, axis=1)
    s["jmerle"] = s.apply(jmerle_mid_row, axis=1)
    s["csv_mid"] = pd.to_numeric(s["mid_price"], errors="coerce")
    s["pnl"] = pd.to_numeric(s["profit_and_loss"], errors="coerce")

    dp = s["pnl"].diff()
    print(f"File: {args.log_json}")
    print(f"Product: {args.product}  rows={len(s)}  timestamps {int(s['timestamp'].iloc[0])}..{int(s['timestamp'].iloc[-1])}")
    print()

    print("corr(ΔPnL, Δcandidate)  — 1.0 = tick-for-tick mark at that series (qty 1)")
    for name in ["micro", "wall", "popular", "jmerle", "csv_mid"]:
        dc = s[name].diff()
        m = dp.notna() & dc.notna()
        c = float(dp[m].corr(dc[m])) if m.sum() > 5 else float("nan")
        print(f"  {name:12s}  {c:+.5f}")

    pnl0 = float(s["pnl"].iloc[0])
    print()
    print("(PnL - PnL0) ~ a + b * (candidate - candidate0)  — want b≈1, low RMSE")
    rows: list[tuple[str, float, float, float]] = []
    for name in ["micro", "wall", "popular", "jmerle"]:
        c0 = float(s[name].iloc[0])
        y = (s["pnl"] - pnl0).to_numpy()
        x = (s[name] - c0).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            continue
        A = np.vstack([np.ones(mask.sum()), x[mask]]).T
        a, b = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        pred = a + b * x[mask]
        rmse = float(np.sqrt(np.mean((y[mask] - pred) ** 2)))
        rows.append((name, b, a, rmse))

    rows.sort(key=lambda t: t[3])
    for name, b, a, rmse in rows:
        print(f"  {name:12s}  b={b:+.5f}  a={a:+.4f}  RMSE={rmse:.4f}")

    print()
    print("corr(ΔPnL, Δrolling_mean(wall, W))  — smoother internal mark?")
    w = s["wall"].to_numpy()
    dpv = dp.to_numpy()
    for W in [3, 5, 10, 20, 30, 50, 80]:
        sw = pd.Series(w).rolling(W, min_periods=W).mean()
        dsw = sw.diff().to_numpy()
        m = np.isfinite(dpv) & np.isfinite(dsw)
        if m.sum() < 20:
            continue
        c = float(np.corrcoef(dpv[m], dsw[m])[0, 1])
        print(f"  W={W:3d}  {c:+.5f}")

    implied = dp.fillna(0.0).cumsum().to_numpy()
    print()
    print("Detrended corr(cumsum(ΔPnL), candidate level)  — path match")
    for name in ["wall", "popular", "micro", "jmerle"]:
        x = s[name].to_numpy()
        c = detrended_corr(implied, x)
        print(f"  {name:12s}  {c:+.4f}")

    print()
    print("Mean |wall - micro|:", float(np.nanmean(np.abs(s["wall"] - s["micro"]))))


if __name__ == "__main__":
    main()
