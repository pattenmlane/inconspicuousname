#!/usr/bin/env python3
"""
After a **one-tick wall-mid spike** (large move from row i-1 → i), did price make
**at least one more move in the same direction** (any strictly positive / negative
row-to-row step, not necessarily large) **before** first re-entry of the
**pre-spike** level ± tol?

Counts separate patterns for **up** then any further **up** before revert, and
**down** then any further **down** before revert. Uses Frankfurt-style wall mid
from L2 CSVs (same as ``analyze_osmium_wall_mid_spikes.py``).

Usage:
  python3 Prosperity4Data/analyze_osmium_wall_mid_spike_chain_before_revert.py
  python3 Prosperity4Data/analyze_osmium_wall_mid_spike_chain_before_revert.py --threshold 3 --tol 1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_osmium_micro_mid_vs_vol_mid import (  # noqa: E402
    PRODUCT,
    ROUND,
    _data_dir,
    load_raw,
)
from analyze_osmium_wall_mid_spikes import wall_mid_row  # noqa: E402


def discover_days(root: Path) -> list[int]:
    days: list[int] = []
    for p in root.glob(f"prices_round_{ROUND}_day_*.csv"):
        m = re.search(r"day_(-?\d+)\.csv$", p.name)
        if m:
            days.append(int(m.group(1)))
    return sorted(days)


def load_wall_all_days(root: Path, days: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[float] = []
    ds: list[int] = []
    ts: list[int] = []
    for day in days:
        df = load_raw(root, day)
        df["wall_mid"] = df.apply(wall_mid_row, axis=1)
        df = df.loc[df["wall_mid"].notna()].copy()
        df["day"] = day
        for _, row in df.iterrows():
            xs.append(float(row["wall_mid"]))
            ds.append(int(row["day"]))
            ts.append(int(row["timestamp"]))
    return (
        np.array(xs, dtype=np.float64),
        np.array(ds, dtype=np.int64),
        np.array(ts, dtype=np.int64),
    )


def scan_patterns(
    x: np.ndarray,
    day: np.ndarray,
    ts: np.ndarray,
    spike_thr: float,
    tol: float,
) -> dict[str, int | float | list]:
    n = len(x)
    up_spikes = 0
    down_spikes = 0
    up_then_nudge_up_before_revert = 0
    down_then_nudge_down_before_revert = 0
    up_censored = 0  # never reverted same day
    down_censored = 0
    up_reverted = 0
    down_reverted = 0
    up_rev_rows: list[int] = []
    up_rev_dt: list[int] = []
    down_rev_rows: list[int] = []
    down_rev_dt: list[int] = []

    for i in range(1, n):
        if day[i] != day[i - 1]:
            continue
        pre = float(x[i - 1])
        d = float(x[i]) - pre
        lo, hi = pre - tol, pre + tol

        if d >= spike_thr:
            up_spikes += 1
            saw_nudge = False
            revert_at = None
            for j in range(i + 1, n):
                if day[j] != day[i]:
                    break
                if x[j] > x[j - 1]:
                    saw_nudge = True
                if lo <= float(x[j]) <= hi:
                    revert_at = j
                    break
            if revert_at is not None:
                up_reverted += 1
                up_rev_rows.append(int(revert_at - i))
                up_rev_dt.append(int(ts[revert_at] - ts[i]))
                if saw_nudge:
                    up_then_nudge_up_before_revert += 1
            else:
                up_censored += 1

        elif d <= -spike_thr:
            down_spikes += 1
            saw_nudge = False
            revert_at = None
            for j in range(i + 1, n):
                if day[j] != day[i]:
                    break
                if x[j] < x[j - 1]:
                    saw_nudge = True
                if lo <= float(x[j]) <= hi:
                    revert_at = j
                    break
            if revert_at is not None:
                down_reverted += 1
                down_rev_rows.append(int(revert_at - i))
                down_rev_dt.append(int(ts[revert_at] - ts[i]))
                if saw_nudge:
                    down_then_nudge_down_before_revert += 1
            else:
                down_censored += 1

    out: dict[str, int | float | list] = {
        "n_rows": n,
        "up_spikes": up_spikes,
        "down_spikes": down_spikes,
        "up_reverted": up_reverted,
        "down_reverted": down_reverted,
        "up_censored": up_censored,
        "down_censored": down_censored,
        "up_then_extra_up_before_revert": up_then_nudge_up_before_revert,
        "down_then_extra_down_before_revert": down_then_nudge_down_before_revert,
        "up_rev_rows": up_rev_rows,
        "up_rev_dt": up_rev_dt,
        "down_rev_rows": down_rev_rows,
        "down_rev_dt": down_rev_dt,
    }
    if up_reverted > 0:
        out["P_up_chain_given_revert"] = up_then_nudge_up_before_revert / up_reverted
    if down_reverted > 0:
        out["P_down_chain_given_revert"] = down_then_nudge_down_before_revert / down_reverted
    if up_spikes > 0:
        out["P_up_chain_given_spike"] = up_then_nudge_up_before_revert / up_spikes
    if down_spikes > 0:
        out["P_down_chain_given_spike"] = down_then_nudge_down_before_revert / down_spikes
    same_day_steps = sum(1 for i in range(1, n) if day[i] == day[i - 1])
    if same_day_steps > 0:
        out["same_day_steps"] = same_day_steps
        out["up_spike_rate"] = up_spikes / same_day_steps
        out["down_spike_rate"] = down_spikes / same_day_steps
    return out


def _summarize_lags(name: str, rows: list[int], dts: list[int]) -> None:
    if not rows:
        print(f"  {name}: (no reverting events)")
        return
    r = np.array(rows, dtype=np.float64)
    t = np.array(dts, dtype=np.float64)
    print(f"  {name} — rows until revert: mean {r.mean():.1f}  median {np.median(r):.0f}  p90 {np.quantile(r, 0.9):.0f}  max {r.max():.0f}")
    print(f"  {name} — Δtimestamp until revert: mean {t.mean():.0f}  median {np.median(t):.0f}  p90 {np.quantile(t, 0.9):.0f}  max {t.max():.0f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Wall mid: spike then same-dir nudge before revert.")
    p.add_argument(
        "--days",
        "-d",
        type=int,
        nargs="*",
        default=None,
        help="Day ids (default: all prices_round_*_day_*.csv under ROUND1).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="First-leg spike: up if Δ>=T, down if Δ<=-T (default 4).",
    )
    p.add_argument("--tol", type=float, default=1.0, help="Revert band around pre-spike level.")
    args = p.parse_args()

    root = _data_dir()
    days = sorted(args.days) if args.days else discover_days(root)
    if not days:
        raise SystemExit("No day CSVs found.")

    x, day, ts = load_wall_all_days(root, days)
    if len(x) < 2:
        raise SystemExit("Not enough rows.")

    s = scan_patterns(x, day, ts, args.threshold, args.tol)

    print(f"{PRODUCT} — wall mid spike → same-direction nudge → revert (pre ± tol)")
    print(f"Days: {days}")
    print(f"Rows with wall_mid: {s['n_rows']}")
    print(f"Spike threshold: up Δ≥{args.threshold}, down Δ≤-{args.threshold}  |  revert: pre ± {args.tol}")
    print()
    print("First-leg spikes (one tick):")
    print(f"  up-spikes:   {s['up_spikes']}")
    print(f"  down-spikes: {s['down_spikes']}")
    if s.get("up_spike_rate") is not None:
        print(
            f"  as % of same-day row-to-row steps ({s['same_day_steps']} steps): "
            f"{s['up_spike_rate']*100:.4f}% up, {s['down_spike_rate']*100:.4f}% down"
        )
    print()
    print("Among up-spikes that later reverted to pre (same day):")
    print(f"  reverted: {s['up_reverted']}  censored (no revert before day end): {s['up_censored']}")
    print(f"  had ≥1 extra UP tick before revert: {s['up_then_extra_up_before_revert']}")
    if s.get("P_up_chain_given_revert") is not None:
        print(f"  P(extra up | reverted): {s['P_up_chain_given_revert']*100:.2f}%")
    if s.get("P_up_chain_given_spike") is not None:
        print(f"  P(extra up | spike):    {s['P_up_chain_given_spike']*100:.2f}%")
    print()
    print("Among down-spikes that later reverted to pre (same day):")
    print(f"  reverted: {s['down_reverted']}  censored: {s['down_censored']}")
    print(f"  had ≥1 extra DOWN tick before revert: {s['down_then_extra_down_before_revert']}")
    if s.get("P_down_chain_given_revert") is not None:
        print(f"  P(extra down | reverted): {s['P_down_chain_given_revert']*100:.2f}%")
    if s.get("P_down_chain_given_spike") is not None:
        print(f"  P(extra down | spike):    {s['P_down_chain_given_spike']*100:.2f}%")

    print()
    print("Time to revert (spike at row i → first row j in pre±tol, same day):")
    total_spikes = int(s["up_spikes"]) + int(s["down_spikes"])
    never = int(s["up_censored"]) + int(s["down_censored"])
    reverted = int(s["up_reverted"]) + int(s["down_reverted"])
    print(f"  All spikes: {total_spikes}  |  reverted: {reverted}  |  never reverted (censored): {never}")
    if total_spikes > 0:
        print(f"  P(never revert | spike): {100.0 * never / total_spikes:.2f}%")
    print()
    _summarize_lags("Up-spikes that reverted", list(s["up_rev_rows"]), list(s["up_rev_dt"]))
    _summarize_lags("Down-spikes that reverted", list(s["down_rev_rows"]), list(s["down_rev_dt"]))
    all_rows = list(s["up_rev_rows"]) + list(s["down_rev_rows"])
    all_dt = list(s["up_rev_dt"]) + list(s["down_rev_dt"])
    print()
    _summarize_lags("Combined (up+down, reverted only)", all_rows, all_dt)


if __name__ == "__main__":
    main()
