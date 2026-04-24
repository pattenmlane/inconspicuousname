#!/usr/bin/env python3
"""
ASH_COATED_OSMIUM — quantity distribution **conditional on extrema touches** vs baseline.

**--extrema trade** (default): same causal flags as ``analyze_osmium_trade_extrema_insider_probe.py``:
``at_low`` / ``at_high`` when trade price is within ``tol`` of **prior** min/max **trade** prices.

**--extrema mid**: for each trade, take the last ``mid_price`` (nonzero) from ``prices_round_*``
with ``timestamp <=`` trade time; compare to **prior** min/max of **mid** over price rows
with strictly earlier timestamps. ``at_low`` / ``at_high`` = mid within ``tol`` of those
running extrema.

**--extrema popular-mid**: same causal logic on **popular (volume) mid** — price at max
displayed size on bid and ask, averaged (see ``plot_osmium_micro_mid_vs_vol_mid.vol_mid_row``).

Reports:
  * Full P(q|slice) for each observed lot size with enrichment vs baseline
  * Even vs odd lot fractions
  * Max single-trade clip and percentiles of quantity per slice

Weak evidence by design — use as a **score**, not a label (no buyer/seller in CSV).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TRADES_DIR = Path(__file__).resolve().parent / "ROUND1"
PRODUCT = "ASH_COATED_OSMIUM"
ROUND = 1


def discover_trade_days() -> list[int]:
    days: list[int] = []
    for p in TRADES_DIR.glob(f"trades_round_{ROUND}_day_*.csv"):
        m = re.search(r"day_(-?\d+)\.csv$", p.name)
        if m:
            days.append(int(m.group(1)))
    return sorted(days)


def load_osmium_trades(day: int) -> pd.DataFrame:
    path = TRADES_DIR / f"trades_round_{ROUND}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["symbol"] == PRODUCT].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    return df.dropna(subset=["quantity"])


def enrich_flags(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    prices: list[float] = []
    at_low: list[bool] = []
    at_high: list[bool] = []
    for _, row in df.iterrows():
        px = float(row["price"])
        if not prices:
            at_low.append(False)
            at_high.append(False)
            prices.append(px)
            continue
        pmin, pmax = min(prices), max(prices)
        at_low.append(px <= pmin + tol)
        at_high.append(px >= pmax - tol)
        prices.append(px)
    out = df.copy()
    out["at_low"] = at_low
    out["at_high"] = at_high
    return out


def load_mid_series(day: int) -> pd.DataFrame:
    path = TRADES_DIR / f"prices_round_{ROUND}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["product"] == PRODUCT, ["timestamp", "mid_price"]].copy()
    df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
    df = df.loc[(df["mid_price"].notna()) & (df["mid_price"] != 0)]
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df = df.reset_index(drop=True)
    return df


def _levels(row: pd.Series, side: str) -> list[tuple[float, float]]:
    """(price, volume) for bid/ask levels with vol > 0 (same as plot_osmium_micro_mid_vs_vol_mid)."""
    out: list[tuple[float, float]] = []
    for i in range(1, 4):
        p = row.get(f"{side}_price_{i}")
        v = row.get(f"{side}_volume_{i}")
        if pd.isna(p) or pd.isna(v):
            continue
        vf = float(v)
        if vf <= 0:
            continue
        out.append((float(p), vf))
    return out


def vol_mid_row(row: pd.Series) -> float | None:
    """Popular mid = (pop bid + pop ask) / 2 from L2 snapshot."""
    bids = _levels(row, "bid")
    asks = _levels(row, "ask")
    if not bids or not asks:
        return None
    popular_bid = max(bids, key=lambda t: t[1])[0]
    popular_ask = max(asks, key=lambda t: t[1])[0]
    return (popular_bid + popular_ask) / 2.0


def load_popular_mid_series(day: int) -> pd.DataFrame:
    path = TRADES_DIR / f"prices_round_{ROUND}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["product"] == PRODUCT].copy()
    df = df.sort_values("timestamp")
    df["popular_mid"] = df.apply(vol_mid_row, axis=1)
    df = df.loc[df["popular_mid"].notna(), ["timestamp", "popular_mid"]].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


def enrich_series_extrema_flags(
    trades: pd.DataFrame, series_df: pd.DataFrame, value_col: str, tol: float
) -> pd.DataFrame:
    """at_low / at_high when series value is within tol of running prior min/max on price timestamps."""
    if series_df.empty:
        o = trades.copy()
        o["at_low"] = False
        o["at_high"] = False
        return o
    ts = series_df["timestamp"].to_numpy(dtype=np.int64)
    m = series_df[value_col].to_numpy(dtype=float)
    prefmin = np.minimum.accumulate(m)
    prefmax = np.maximum.accumulate(m)
    at_low: list[bool] = []
    at_high: list[bool] = []
    for _, row in trades.iterrows():
        t = int(row["timestamp"])
        idx = int(np.searchsorted(ts, t, side="right")) - 1
        if idx < 0:
            at_low.append(False)
            at_high.append(False)
            continue
        v_now = float(m[idx])
        if idx == 0:
            at_low.append(False)
            at_high.append(False)
            continue
        prior_min = float(prefmin[idx - 1])
        prior_max = float(prefmax[idx - 1])
        at_low.append(v_now <= prior_min + tol)
        at_high.append(v_now >= prior_max - tol)
    out = trades.copy()
    out["at_low"] = at_low
    out["at_high"] = at_high
    return out


def pmf_counts(q: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique quantities sorted, counts)."""
    q = q.astype(int)
    if q.empty:
        return np.array([], dtype=int), np.array([], dtype=int)
    u, c = np.unique(q.to_numpy(), return_counts=True)
    return u, c


def p_at(u: np.ndarray, c: np.ndarray, k: int) -> float:
    tot = int(c.sum())
    if tot == 0:
        return float("nan")
    idx = np.searchsorted(u, k)
    if idx < len(u) and u[idx] == k:
        return float(c[idx]) / tot
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1.0, help="Ticks from prior extrema (price units).")
    ap.add_argument(
        "--extrema",
        choices=("trade", "mid", "popular-mid"),
        default="trade",
        help="Extrema: prior trade prices, logged mid_price, or popular (vol) mid from L2.",
    )
    args = ap.parse_args()

    days = discover_trade_days()
    frames: list[pd.DataFrame] = []
    for d in days:
        t = load_osmium_trades(d)
        if t.empty:
            continue
        t["day"] = d
        if args.extrema == "trade":
            t = enrich_flags(t, args.tol)
        elif args.extrema == "mid":
            mids = load_mid_series(d)
            t = enrich_series_extrema_flags(t, mids, "mid_price", args.tol)
        else:
            pop = load_popular_mid_series(d)
            t = enrich_series_extrema_flags(t, pop, "popular_mid", args.tol)
        frames.append(t)
    if not frames:
        raise SystemExit("No osmium trades.")

    df = pd.concat(frames, ignore_index=True)
    base_mask = ~(df["at_low"] | df["at_high"])
    low_mask = df["at_low"]
    high_mask = df["at_high"]

    q_lo = df.loc[low_mask, "quantity"].astype(int)
    q_hi = df.loc[high_mask, "quantity"].astype(int)
    q_b = df.loc[base_mask, "quantity"].astype(int)

    n = len(df)
    n_lo, n_hi, n_b = len(q_lo), len(q_hi), len(q_b)

    ext_labels = {
        "trade": "prior **trade-price** extrema",
        "mid": "prior **mid_price** extrema",
        "popular-mid": "prior **popular (vol) mid** extrema",
    }
    ext_label = ext_labels[args.extrema]
    print(f"{PRODUCT} — quantity vs {ext_label} (pooled days)  [--extrema {args.extrema}]")
    print(f"Days: {days}  trades={n}  tol={args.tol}")
    print(f"  at_low:   n={n_lo:5d}  ({100 * n_lo / n:.2f}%)")
    print(f"  at_high:  n={n_hi:5d}  ({100 * n_hi / n:.2f}%)")
    print(f"  baseline: n={n_b:5d}  ({100 * n_b / n:.2f}%)  (not at_low ∪ at_high)")
    print()

    # --- Even / odd ---
    def even_odd(q: pd.Series) -> tuple[float, float]:
        if q.empty:
            return float("nan"), float("nan")
        x = (q.astype(int) % 2 == 0).to_numpy()
        return float(x.mean()), float((~x).mean())

    el, ol = even_odd(q_lo)
    eh, oh = even_odd(q_hi)
    eb, ob = even_odd(q_b)
    print("--- Even vs odd lot (single-trade quantity) ---")
    print(f"  {'slice':12}  P(even)  P(odd)")
    print(f"  {'at_low':12}  {el:7.3f}  {ol:7.3f}")
    print(f"  {'at_high':12}  {eh:7.3f}  {oh:7.3f}")
    print(f"  {'baseline':12}  {eb:7.3f}  {ob:7.3f}")
    if not np.isnan(el) and not np.isnan(eb) and eb > 0:
        print(f"  P(even|at_low)/P(even|baseline) = {el / eb:.3f}  (same for high: {eh / eb:.3f})  (≈1 if no shift)")
    print()

    # --- Max clip & percentiles ---
    print("--- Max clip & quantity percentiles (per trade = one clip) ---")
    for name, s in [("at_low", q_lo), ("at_high", q_hi), ("baseline", q_b)]:
        if s.empty:
            print(f"  {name}: (empty)")
            continue
        a = s.to_numpy(dtype=int)
        print(
            f"  {name:10}  n={len(a):5d}  max={int(a.max()):3d}  "
            f"p50={float(np.percentile(a, 50)):.1f}  p90={float(np.percentile(a, 90)):.1f}  "
            f"p95={float(np.percentile(a, 95)):.1f}  mean={float(a.mean()):.2f}"
        )
    print()

    # --- Full PMF enrichment for every qty that appears anywhere with meaningful mass ---
    u_all = np.unique(np.concatenate([q_lo.to_numpy(), q_hi.to_numpy(), q_b.to_numpy()]))
    u_lo, c_lo = pmf_counts(q_lo)
    u_hi, c_hi = pmf_counts(q_hi)
    u_b, c_b = pmf_counts(q_b)

    print("--- P(qty=k | slice) vs baseline & enrichment ---")
    print(f"  {'k':>4}  {'P(k|low)':>9}  {'P(k|hi)':>9}  {'P(k|base)':>10}  {'×low':>7}  {'×high':>7}  n_lo n_hi n_b")
    rows: list[tuple] = []
    for k in u_all:
        pl = p_at(u_lo, c_lo, int(k))
        ph = p_at(u_hi, c_hi, int(k))
        pb = p_at(u_b, c_b, int(k))
        nl = int((q_lo == k).sum()) if n_lo else 0
        nh = int((q_hi == k).sum()) if n_hi else 0
        nb = int((q_b == k).sum()) if n_b else 0
        rl = pl / pb if pb > 0 and not np.isnan(pl) else float("nan")
        rh = ph / pb if pb > 0 and not np.isnan(ph) else float("nan")
        rows.append((k, pl, ph, pb, rl, rh, nl, nh, nb))

    # Print rows where any slice has at least 1% mass or enrichment notable, or always print all with n_b>=3
    for k, pl, ph, pb, rl, rh, nl, nh, nb in rows:
        if max(pl, ph, pb, nl / max(n_lo, 1), nh / max(n_hi, 1), nb / max(n_b, 1)) < 0.005 and nl + nh + nb < 15:
            continue
        rls = f"{rl:.2f}" if np.isfinite(rl) else ("inf" if pb == 0 and nl > 0 else "—")
        rhs = f"{rh:.2f}" if np.isfinite(rh) else ("inf" if pb == 0 and nh > 0 else "—")
        print(
            f"  {int(k):4d}  {pl:9.4f}  {ph:9.4f}  {pb:10.4f}  {rls:>7}  {rhs:>7}  {nl:4d} {nh:4d} {nb:4d}"
        )

    print()
    print("  (enrichment × = P(k|slice)/P(k|baseline); inf = baseline never uses that k)")
    print()
    if args.extrema == "trade":
        print(
            "Note: first trade each day has no prior trade extrema — excluded from at_low/at_high "
            "by construction. Interpret pooled enrichment cautiously (thin tails on rare k)."
        )
    elif args.extrema == "mid":
        print(
            "Note (--extrema mid): trades need ≥2 mid snapshots through merge time — first price row "
            "cannot produce a touch. Interpret pooled enrichment cautiously (thin tails on rare k)."
        )
    else:
        print(
            "Note (--extrema popular-mid): same as mid — need ≥2 valid popular_mid rows before the "
            "merge index. Interpret pooled enrichment cautiously (thin tails on rare k)."
        )


if __name__ == "__main__":
    main()
