#!/usr/bin/env python3
"""
Jmerle-style **smoothed rolling z** on ASH_COATED_OSMIUM (same formula as
``analyze_osmium_jmerle_style_signal`` / squid ink notebook):

    z_raw = (mid - rolling_mean(mid, Wz)) / rolling_std(mid, Wz)
    signal = rolling_mean(z_raw, Ws)

Then offline stats to see if z-scores **predict short-horizon mean reversion**
(negative correlation of signal with forward mid changes = fade-the-stretch).

**Mid choices** (``--mid``):
  * ``vol`` — max-volume bid + max-volume ask / 2 (Prosperity4Data ``vol_mid_row``)
  * ``jmerle`` — jmerle ink: max-vol **bid** + **min**-vol **ask** (see ``INK_INFO/jmerle.py``)
  * ``wall`` — min bid + max ask / 2 (Hedgehogs wall mid)
  * ``micro`` — best bid + best ask / 2
  * ``csv_mid`` — ``mid_price`` column (drops 0)

Forward moves are computed **within each day** (no cross-day ``shift``).

Usage:
  python3 Prosperity4Data/analyze_osmium_zscore_meanrev.py
  python3 Prosperity4Data/analyze_osmium_zscore_meanrev.py --mid jmerle --windows 20,15 30,20 150,100
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_osmium_wall_mid_spikes import wall_mid_row  # noqa: E402
from plot_osmium_micro_mid_vs_vol_mid import (  # noqa: E402
    PRODUCT,
    ROUND,
    _levels,
    load_raw,
    micro_mid_row,
    vol_mid_row,
)


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / f"ROUND{ROUND}"


def discover_days(root: Path) -> list[int]:
    days: list[int] = []
    for p in root.glob(f"prices_round_{ROUND}_day_*.csv"):
        m = re.search(r"day_(-?\d+)\.csv$", p.name)
        if m:
            days.append(int(m.group(1)))
    return sorted(days)


def jmerle_pop_mid_row(row: pd.Series) -> float | None:
    """Squid ink / jmerle: largest bid size level + smallest ask size level."""
    bids = _levels(row, "bid")
    asks = _levels(row, "ask")
    if not bids or not asks:
        return None
    popular_buy = max(bids, key=lambda t: t[1])[0]
    popular_sell = min(asks, key=lambda t: t[1])[0]
    return (popular_buy + popular_sell) / 2.0


def mid_series(root: Path, days: list[int], mid_kind: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in days:
        df = load_raw(root, d)
        if mid_kind == "micro":
            df["m"] = df.apply(micro_mid_row, axis=1)
        elif mid_kind == "vol":
            df["m"] = df.apply(vol_mid_row, axis=1)
        elif mid_kind == "wall":
            df["m"] = df.apply(wall_mid_row, axis=1)
        elif mid_kind == "jmerle":
            df["m"] = df.apply(jmerle_pop_mid_row, axis=1)
        elif mid_kind == "csv_mid":
            df["m"] = pd.to_numeric(df["mid_price"], errors="coerce")
            df.loc[df["m"] == 0, "m"] = np.nan
        else:
            raise ValueError(mid_kind)
        df = df.loc[df["m"].notna()].copy()
        df["day"] = d
        frames.append(df[["day", "m"]])
    out = pd.concat(frames, ignore_index=True)
    return out


def smoothed_z(mid: pd.Series, wz: int, ws: int) -> pd.Series:
    rmean = mid.rolling(wz, min_periods=wz).mean()
    rstd = mid.rolling(wz, min_periods=wz).std()
    z = (mid - rmean) / rstd
    return z.rolling(ws, min_periods=ws).mean()


def add_forward_by_day(df: pd.DataFrame, horizons: list[int]) -> None:
    g = df.groupby("day", sort=False)["m"]
    for h in horizons:
        df[f"f{h}"] = g.transform(lambda s: s.shift(-h) - s)


def analyze_one(df: pd.DataFrame, wz: int, ws: int, thresh: float) -> None:
    df = df.copy()
    df["sig"] = df.groupby("day", sort=False)["m"].transform(lambda s: smoothed_z(s, wz, ws))
    horizons = [1, 3, 5, 10, 20]
    add_forward_by_day(df, horizons)

    v = df["sig"].notna()
    for h in horizons:
        v &= df[f"f{h}"].notna()
    sub = df.loc[v]

    if len(sub) < 50:
        print(f"  Wz={wz} Ws={ws}: too few valid rows ({len(sub)})")
        return

    print(f"  Wz={wz} Ws={ws}  valid rows={len(sub)}")
    for h in horizons:
        c = sub["sig"].corr(sub[f"f{h}"])
        print(f"    corr(sig, f{h}): {c:+.4f}  (negative => MR: high sig → mid falls)")

    hi = sub["sig"] > thresh
    lo = sub["sig"] < -thresh
    midn = sub["sig"].abs() < 0.25
    print(f"    mean f1 | sig>{thresh}: {sub.loc[hi, 'f1'].mean():+.4f}  (n={hi.sum()})")
    print(f"    mean f1 | sig<-{thresh}: {sub.loc[lo, 'f1'].mean():+.4f}  (n={lo.sum()})")
    print(f"    mean f1 | |sig|<0.25: {sub.loc[midn, 'f1'].mean():+.4f}  (n={midn.sum()})")

    # Lag-1 autocorr of signal (pooled within-day is biased; use per-day ac then mean)
    acs: list[float] = []
    for _, g in df.groupby("day"):
        s = g["sig"].dropna().to_numpy(dtype=np.float64)
        if len(s) > 5:
            acs.append(float(np.corrcoef(s[1:], s[:-1])[0, 1]))
    if acs:
        print(f"    mean(acf(sig,1) per day): {float(np.nanmean(acs)):+.4f}")


def parse_windows(specs: list[str]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s in specs:
        parts = s.replace(" ", "").split(",")
        if len(parts) != 2:
            raise ValueError(f"Bad --windows entry {s!r}; use Wz,Ws e.g. 50,30")
        out.append((int(parts[0]), int(parts[1])))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Z-score mean-reversion stats (jmerle-style).")
    p.add_argument("--days", "-d", type=int, nargs="*", default=None, help="Day ids (default: all CSVs)")
    p.add_argument(
        "--mid",
        choices=("vol", "jmerle", "wall", "micro", "csv_mid"),
        default="vol",
        help="Price series for z-score (default vol = max-vol bid+ask).",
    )
    p.add_argument(
        "--windows",
        "-w",
        nargs="+",
        default=["20,15", "30,20", "50,30", "80,50", "150,100"],
        help="Pairs Wz,Ws (rolling z window, smooth window).",
    )
    p.add_argument("--thresh", type=float, default=1.0, help="High/low signal cut like jmerle ±1.")
    args = p.parse_args()

    root = _data_dir()
    days = sorted(args.days) if args.days else discover_days(root)
    if not days:
        raise SystemExit("No day CSVs.")

    df = mid_series(root, days, args.mid)
    print(f"{PRODUCT} — smoothed rolling z mean-reversion probe")
    print(f"Days: {days}  rows: {len(df)}  mid={args.mid!r}")
    print(f"Threshold for hi/lo buckets: ±{args.thresh}")
    print()

    windows = parse_windows(args.windows)
    for wz, ws in windows:
        if wz < 2 or ws < 1:
            continue
        analyze_one(df, wz, ws, args.thresh)
        print()

    print("--- How to read ---")
    print("If strategy is 'fade' (long when sig<<0): you want mean f1 | sig<-1 to be **positive**")
    print("(mid rises next tick) and mean f1 | sig>+1 to be **negative**.")
    print("corr(sig, f1) **negative** is the same story in one number.")


if __name__ == "__main__":
    main()
