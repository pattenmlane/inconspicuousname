#!/usr/bin/env python3
"""
Exploratory plot in the spirit of INK_INFO/round1.ipynb (jmerle squid ink):

  - Mid price time series for ASH_COATED_OSMIUM
  - "Signal" line: ((mid - rolling(Wz).mean()) / rolling(Wz).std()).rolling(Ws).mean()
    (same form as the notebook, which used Wz=50, Ws=100; jmerle's *trader* used 150/100.)

Uses Prosperity4Data prices_round_1_day_*.csv (semicolon). Drops mid_price == 0.

Usage:
  python3 Prosperity4Data/analyze_osmium_jmerle_style_signal.py --days -2 -1 0
  python3 Prosperity4Data/analyze_osmium_jmerle_style_signal.py -d -2 --wz 50 --ws 100 --no-show

Outputs PNG next to the data under ROUND1/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PRODUCT = "ASH_COATED_OSMIUM"
ROUND = 1


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / f"ROUND{ROUND}"


def load_mids(root: Path, days: list[int], two_sided: bool) -> pd.DataFrame:
    frames = []
    for day in days:
        path = root / f"prices_round_{ROUND}_day_{day}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, sep=";")
        df = df.loc[df["product"] == PRODUCT].copy()
        df = df.sort_values("timestamp")
        df = df.loc[(df["mid_price"].notna()) & (df["mid_price"] != 0)]
        if two_sided:
            df = df.loc[df["bid_price_1"].notna() & df["ask_price_1"].notna()]
        df["day"] = day
        frames.append(df[["day", "timestamp", "mid_price"]])
    out = pd.concat(frames, ignore_index=True)
    out["i"] = range(len(out))
    return out


def smoothed_z(mid: pd.Series, wz: int, ws: int) -> pd.Series:
    rmean = mid.rolling(wz, min_periods=wz).mean()
    rstd = mid.rolling(wz, min_periods=wz).std()
    z = (mid - rmean) / rstd
    return z.rolling(ws, min_periods=ws).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", "-d", type=int, nargs="+", default=[-2], help="Day ids (default: -2)")
    p.add_argument("--wz", type=int, default=50, help="Z-score rolling window (notebook used 50; trader used 150)")
    p.add_argument("--ws", type=int, default=100, help="Smoothing rolling window on z")
    p.add_argument(
        "--two-sided-only",
        action="store_true",
        help="Require bid_price_1 and ask_price_1 (cleaner mids)",
    )
    p.add_argument("--slice-start", type=int, default=None, help="Optional row start index (like notebook iloc)")
    p.add_argument("--slice-stop", type=int, default=None, help="Optional row stop index exclusive")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    root = _data_dir()
    df = load_mids(root, args.days, args.two_sided_only)
    if args.slice_start is not None or args.slice_stop is not None:
        lo = args.slice_start or 0
        hi = args.slice_stop or len(df)
        df = df.iloc[lo:hi].copy()
        df["i"] = range(len(df))

    mid = df["mid_price"]
    sig = smoothed_z(mid, args.wz, args.ws)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df["i"], mid, color="#4a4a8c", lw=0.8, label="mid")
    axes[0].set_ylabel("mid_price")
    axes[0].set_title(f"{PRODUCT} — mid (days {args.days})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(df["i"], sig, color="#8c4a4a", lw=0.9, label="smoothed z")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].axhline(1, color="g", ls="--", lw=0.6, alpha=0.7)
    axes[1].axhline(-1, color="g", ls="--", lw=0.6, alpha=0.7)
    axes[1].set_ylabel("signal")
    axes[1].set_xlabel("row index (concatenated days)")
    axes[1].set_title(f"Signal: roll_z(mid,{args.wz}) then roll_mean(,{args.ws})  (±1 thresholds like jmerle)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    tag = "_".join(map(str, args.days))
    out = root / f"analysis_osmium_jmerle_signal_r{ROUND}_days{tag}_wz{args.wz}_ws{args.ws}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}  (n={len(df)})")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
