#!/usr/bin/env python3
"""
Spike / mean-reversion stats for ASH_COATED_OSMIUM **popular mid**:
(max-volume bid price + max-volume ask price) / 2 from L2 CSVs
(same definition as ``plot_osmium_micro_mid_vs_vol_mid.vol_mid_row``).

For each row *i* with a large step |pop_mid[i] - pop_mid[i-1]| >= threshold,
records pre-spike level (pop at i-1) and measures how long until popular mid
first returns within ``tol`` of that level (full reversion), and optionally
time to 50%% reversion of the step.

Uses ``prices_round_1_day_<n>.csv`` under ``Prosperity4Data/ROUND1/``.

Usage:
  python3 Prosperity4Data/analyze_osmium_popular_mid_spikes.py --days -2 -1 0
  python3 Prosperity4Data/analyze_osmium_popular_mid_spikes.py -d -2 --threshold 5 --tol 1.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Same folder as this script → import sibling module by path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_osmium_micro_mid_vs_vol_mid import (  # noqa: E402
    PRODUCT,
    ROUND,
    _data_dir,
    load_raw,
    vol_mid_row,
)


def load_pop_series(root: Path, days: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in days:
        path = root / f"prices_round_{ROUND}_day_{day}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = load_raw(root, day)
        df["pop_mid"] = df.apply(vol_mid_row, axis=1)
        df = df.loc[df["pop_mid"].notna()].copy()
        df["day"] = day
        frames.append(df[["day", "timestamp", "pop_mid"]])
    out = pd.concat(frames, ignore_index=True)
    out["dt"] = out["timestamp"].diff().fillna(0).astype(np.int64)
    return out


def analyze_spikes(
    df: pd.DataFrame,
    threshold: float,
    tol: float,
    track_half: bool,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Spike events on pop_mid; reversion measured in rows and timestamp delta."""
    pop = df["pop_mid"].to_numpy(dtype=np.float64)
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    n = len(pop)
    day = df["day"].to_numpy()
    d = np.abs(np.diff(pop))
    raw_spike = np.where(d >= threshold)[0] + 1  # i where step from i-1→i is large
    # Ignore first row of each day (concatenated multi-day series has no real "step")
    spike_idx = np.array([i for i in raw_spike if i > 0 and day[i] == day[i - 1]], dtype=np.int64)

    rows: list[dict[str, float | int]] = []
    rev_rows: list[int] = []
    rev_dt: list[int] = []
    half_rows: list[int] = []
    half_dt: list[int] = []

    for i in spike_idx:
        if i <= 0 or i >= n:
            continue
        pre = float(pop[i - 1])
        post = float(pop[i])
        step = post - pre
        lo, hi = pre - tol, pre + tol
        t0 = int(ts[i])

        # full reversion: first j > i with pop[j] in [pre-tol, pre+tol]
        full_r = None
        full_dt = None
        for j in range(i + 1, n):
            if lo <= float(pop[j]) <= hi:
                full_r = j - i
                full_dt = int(ts[j]) - t0
                break

        hr = hdt = None
        if track_half and step != 0.0:
            mid = pre + 0.5 * step
            if step > 0:
                # up spike: half reversion = first j with pop[j] <= mid
                for j in range(i + 1, n):
                    if float(pop[j]) <= mid:
                        hr = j - i
                        hdt = int(ts[j]) - t0
                        break
            else:
                for j in range(i + 1, n):
                    if float(pop[j]) >= mid:
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
    p = argparse.ArgumentParser(description=f"Popular-mid spike stats for {PRODUCT}.")
    p.add_argument("--days", "-d", type=int, nargs="+", default=[-2, -1, 0], help="Day ids")
    p.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Min |Δ pop_mid| between consecutive rows to count as spike (default 4).",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1.0,
        help="Band around pre-spike level for 'full reversion' (default 1).",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="Also report time to 50%% reversion of the step toward pre.",
    )
    p.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional path to write per-spike CSV.",
    )
    args = p.parse_args()

    root = _data_dir()
    df = load_pop_series(root, args.days)
    if df.empty:
        raise SystemExit("No rows with valid popular mid.")

    ev, summary = analyze_spikes(df, args.threshold, args.tol, args.half)

    same_day = df["day"].eq(df["day"].shift(1))
    d1 = df["pop_mid"].diff().abs().loc[same_day.fillna(False)]
    print(f"{PRODUCT} — popular mid (max-vol bid + max-vol ask) / 2")
    print(f"Days: {args.days}  |  rows with pop_mid: {len(df)}")
    print()
    print("Step size |Δpop| (consecutive rows):")
    for q in (0.5, 0.9, 0.99, 0.999):
        print(f"  quantile {q:.3f}: {d1.quantile(q):.4f}")
    print(f"  mean |Δ|: {d1.mean():.4f}  max: {d1.max():.4f}")
    print()
    for k in (2, 3, 4, 5, 6):
        frac = float((d1 >= k).mean())
        print(f"  P(|Δ| >= {k}) = {frac*100:.3f}% of steps")
    print()
    print(f"Spike threshold: |Δ| >= {args.threshold}  reversion tol: ±{args.tol} around pre level")
    print(f"Spikes: {summary['n_spikes']}  ({summary['spike_rate']*100:.3f}% of steps)")
    print(f"Reverted to pre (within tol) before series end: {summary['reverted_count']}")
    print(f"Censored (never re-entered band): {summary['censored_full']}")
    if summary.get("median_rev_rows") is not None:
        print()
        print("Among reverting spikes (time to first touch of pre ± tol):")
        print(f"  median: {summary['median_rev_rows']:.0f} rows, Δt {summary['median_rev_dt']:.0f}")
        print(
            f"  p75 / p90 / p95: rows {summary['p75_rev_rows']:.0f} / {summary['p90_rev_rows']:.0f} / {summary['p95_rev_rows']:.0f}"
        )
        print(
            f"                   Δt  {summary['p75_rev_dt']:.0f} / {summary['p90_rev_dt']:.0f} / {summary['p95_rev_dt']:.0f}"
        )
        print(
            f"  mean (heavy-tail): {summary['mean_rev_rows']:.0f} rows, Δt {summary['mean_rev_dt']:.0f} — use medians for 'typical'"
        )
    if args.half and summary.get("median_half_rows") is not None:
        print()
        print("50% reversion (toward pre):")
        print(f"  median rows: {summary['median_half_rows']:.1f}  median Δt: {summary['median_half_dt']:.0f}")

    med_dt = float(df.loc[df["dt"] > 0, "dt"].median()) if (df["dt"] > 0).any() else float("nan")
    print()
    print(f"Median Δtimestamp between consecutive rows (same day chain): {med_dt:.1f}")

    if args.csv_out and not ev.empty:
        outp = Path(args.csv_out)
        ev.to_csv(outp, index=False)
        print(f"Wrote spike table: {outp}")


if __name__ == "__main__":
    main()
