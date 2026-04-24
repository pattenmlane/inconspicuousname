#!/usr/bin/env python3
"""
Spike / mean-reversion stats for ASH_COATED_OSMIUM **wall mid** (Frankfurt
Hedgehogs style from ``INK_INFO/FrankfurtHedgehogs_polished.py``):

  bid_wall = lowest bid price with positive displayed volume
  ask_wall = highest ask price with positive displayed volume
  wall_mid = (bid_wall + ask_wall) / 2

Uses the same L2 columns as ``plot_osmium_micro_mid_vs_vol_mid`` (up to 3
levels per side). Same spike / reversion machinery as
``analyze_osmium_popular_mid_spikes.py``.

Usage:
  python3 Prosperity4Data/analyze_osmium_wall_mid_spikes.py --days -2 -1 0
  python3 Prosperity4Data/analyze_osmium_wall_mid_spikes.py -d -2 --threshold 4 --half

On products with only a few L2 rows (e.g. three per side in the price CSV),
``wall_mid`` often coincides with ``popular`` (max-vol) mid; run both scripts
to compare if your book is deeper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_osmium_micro_mid_vs_vol_mid import (  # noqa: E402
    PRODUCT,
    ROUND,
    _data_dir,
    _levels,
    load_raw,
)


def wall_mid_row(row: pd.Series) -> float | None:
    bids = _levels(row, "bid")
    asks = _levels(row, "ask")
    if not bids or not asks:
        return None
    bid_wall = min(p for p, _ in bids)
    ask_wall = max(p for p, _ in asks)
    return (bid_wall + ask_wall) / 2.0


def load_wall_series(root: Path, days: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in days:
        path = root / f"prices_round_{ROUND}_day_{day}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = load_raw(root, day)
        df["wall_mid"] = df.apply(wall_mid_row, axis=1)
        df = df.loc[df["wall_mid"].notna()].copy()
        df["day"] = day
        frames.append(df[["day", "timestamp", "wall_mid"]])
    out = pd.concat(frames, ignore_index=True)
    out["dt"] = out["timestamp"].diff().fillna(0).astype(np.int64)
    return out


def analyze_spikes(
    df: pd.DataFrame,
    col: str,
    threshold: float,
    tol: float,
    track_half: bool,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    x = df[col].to_numpy(dtype=np.float64)
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    n = len(x)
    day = df["day"].to_numpy()
    d = np.abs(np.diff(x))
    raw_spike = np.where(d >= threshold)[0] + 1
    spike_idx = np.array([i for i in raw_spike if i > 0 and day[i] == day[i - 1]], dtype=np.int64)

    rows: list[dict[str, float | int]] = []
    rev_rows: list[int] = []
    rev_dt: list[int] = []
    half_rows: list[int] = []
    half_dt: list[int] = []

    for i in spike_idx:
        if i <= 0 or i >= n:
            continue
        pre = float(x[i - 1])
        post = float(x[i])
        step = post - pre
        lo, hi = pre - tol, pre + tol
        t0 = int(ts[i])

        full_r = full_dt = None
        for j in range(i + 1, n):
            if lo <= float(x[j]) <= hi:
                full_r = j - i
                full_dt = int(ts[j]) - t0
                break

        hr = hdt = None
        if track_half and step != 0.0:
            mid = pre + 0.5 * step
            if step > 0:
                for j in range(i + 1, n):
                    if float(x[j]) <= mid:
                        hr = j - i
                        hdt = int(ts[j]) - t0
                        break
            else:
                for j in range(i + 1, n):
                    if float(x[j]) >= mid:
                        hr = j - i
                        hdt = int(ts[j]) - t0
                        break

        rows.append(
            {
                "i": i,
                "day": int(df["day"].iloc[i]),
                "timestamp": t0,
                "pre": pre,
                "post": post,
                "step": step,
                "abs_step": abs(step),
                "rev_rows": full_r if full_r is not None else -1,
                "rev_dt": full_dt if full_dt is not None else -1,
                "half_rows": hr if hr is not None else -1,
                "half_dt": hdt if hdt is not None else -1,
            }
        )
        if full_r is not None:
            rev_rows.append(full_r)
            rev_dt.append(full_dt)
        if track_half and hr is not None:
            half_rows.append(hr)
            half_dt.append(hdt)

    ev = pd.DataFrame(rows)
    total_steps = max(n - 1, 0)
    n_spike = len(ev)
    summary: dict[str, float | int] = {
        "rows": n,
        "total_steps": total_steps,
        "n_spikes": n_spike,
        "spike_rate": (n_spike / total_steps) if total_steps else 0.0,
        "reverted_count": len(rev_rows),
        "censored_full": n_spike - len(rev_rows),
    }
    if rev_rows:
        arr_r = np.array(rev_rows, dtype=np.float64)
        arr_t = np.array(rev_dt, dtype=np.float64)
        summary["median_rev_rows"] = float(np.median(arr_r))
        summary["median_rev_dt"] = float(np.median(arr_t))
        summary["mean_rev_rows"] = float(np.mean(arr_r))
        summary["mean_rev_dt"] = float(np.mean(arr_t))
        for q in (0.75, 0.9, 0.95):
            summary[f"p{int(q*100)}_rev_rows"] = float(np.quantile(arr_r, q))
            summary[f"p{int(q*100)}_rev_dt"] = float(np.quantile(arr_t, q))
    if track_half and half_rows:
        summary["median_half_rows"] = float(np.median(half_rows))
        summary["median_half_dt"] = float(np.median(half_dt))

    return ev, summary


def main() -> None:
    p = argparse.ArgumentParser(description=f"Wall-mid spike stats for {PRODUCT}.")
    p.add_argument("--days", "-d", type=int, nargs="+", default=[-2, -1, 0], help="Day ids")
    p.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Min |Δ wall_mid| between consecutive rows to count as spike (default 4).",
    )
    p.add_argument("--tol", type=float, default=1.0, help="Band around pre for full reversion.")
    p.add_argument("--half", action="store_true", help="Report 50%% reversion times.")
    p.add_argument("--csv-out", type=str, default=None, help="Write per-spike CSV.")
    args = p.parse_args()

    root = _data_dir()
    df = load_wall_series(root, args.days)
    if df.empty:
        raise SystemExit("No rows with valid wall_mid.")

    ev, summary = analyze_spikes(df, "wall_mid", args.threshold, args.tol, args.half)

    same_day = df["day"].eq(df["day"].shift(1))
    d1 = df["wall_mid"].diff().abs().loc[same_day.fillna(False)]

    print(f"{PRODUCT} — wall mid (min bid px + max ask px) / 2  [Hedgehogs get_walls]")
    print(f"Days: {args.days}  |  rows with wall_mid: {len(df)}")
    print()
    print("Step size |Δwall| (consecutive rows, same day):")
    for q in (0.5, 0.9, 0.99, 0.999):
        print(f"  quantile {q:.3f}: {d1.quantile(q):.4f}")
    print(f"  mean |Δ|: {d1.mean():.4f}  max: {d1.max():.4f}")
    print()
    for k in (2, 4, 6, 8, 10, 12, 16):
        frac = float((d1 >= k).mean())
        print(f"  P(|Δ| >= {k}) = {frac*100:.3f}% of steps")
    print()
    print(f"Spike threshold: |Δ| >= {args.threshold}  reversion tol: ±{args.tol}")
    print(f"Spikes: {summary['n_spikes']}  ({summary['spike_rate']*100:.3f}% of steps)")
    print(f"Reverted to pre (within tol): {summary['reverted_count']}")
    print(f"Censored: {summary['censored_full']}")
    if summary.get("median_rev_rows") is not None:
        print()
        print("Among reverting spikes:")
        print(f"  median: {summary['median_rev_rows']:.0f} rows, Δt {summary['median_rev_dt']:.0f}")
        print(
            f"  p75 / p90 / p95: rows {summary['p75_rev_rows']:.0f} / {summary['p90_rev_rows']:.0f} / {summary['p95_rev_rows']:.0f}"
        )
        print(
            f"                   Δt  {summary['p75_rev_dt']:.0f} / {summary['p90_rev_dt']:.0f} / {summary['p95_rev_dt']:.0f}"
        )
        print(f"  mean: {summary['mean_rev_rows']:.0f} rows, Δt {summary['mean_rev_dt']:.0f}")
    if args.half and summary.get("median_half_rows") is not None:
        print()
        print("50% reversion:")
        print(f"  median rows: {summary['median_half_rows']:.1f}  median Δt: {summary['median_half_dt']:.0f}")

    med_dt = float(df.loc[df["dt"] > 0, "dt"].median()) if (df["dt"] > 0).any() else float("nan")
    print()
    print(f"Median Δtimestamp between consecutive rows: {med_dt:.1f}")

    if args.csv_out and not ev.empty:
        outp = Path(args.csv_out)
        ev.to_csv(outp, index=False)
        print(f"Wrote: {outp}")


if __name__ == "__main__":
    main()
