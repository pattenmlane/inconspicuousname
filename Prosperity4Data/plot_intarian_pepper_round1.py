#!/usr/bin/env python3
"""
Plot INTARIAN_PEPPER_ROOT mid price for Prosperity 4 Round 1 for one day.
Y (and X) axes are tight to the min/max of the series for that day.
Rows with mid_price == 0 are dropped (empty book in the log).

Usage:
  python3 Prosperity4Data/plot_intarian_pepper_round1.py --day -2
  python3 Prosperity4Data/plot_intarian_pepper_round1.py -d 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PRODUCT = "INTARIAN_PEPPER_ROOT"
ROUND = 1


def _data_dir() -> Path:
    here = Path(__file__).resolve().parent
    return here / f"ROUND{ROUND}"


def load_day(root: Path, day: int) -> pd.DataFrame:
    path = root / f"prices_round_{ROUND}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["product"] == PRODUCT].copy()
    df = df.sort_values("timestamp")
    # No bids/asks → mid_price 0 in CSV; omit so the plot and y-limits stay meaningful
    df = df.loc[(df["mid_price"].notna()) & (df["mid_price"] != 0)].copy()
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=f"Plot {PRODUCT} mid price (Round {ROUND}).")
    p.add_argument(
        "--day",
        "-d",
        type=int,
        default=-2,
        help="Trading day (default: -2). Must match prices_round_*_day_<n>.csv on disk.",
    )
    args = p.parse_args()
    day = args.day

    root = _data_dir()
    if not root.is_dir():
        raise SystemExit(f"Missing data folder: {root}")

    df = load_day(root, day)
    if df.empty:
        raise SystemExit(
            f"No plottable rows for {PRODUCT} on day {day} (missing file, no rows, or all mid_price == 0)."
        )

    ts = df["timestamp"]
    mid = df["mid_price"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, mid, linewidth=0.9, color="#2a6f3b")
    ax.set_title(f"{PRODUCT} — Round {ROUND}, day {day} (mid price, tight axes)")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("mid_price")
    ax.grid(True, alpha=0.35)

    ymin, ymax = float(mid.min()), float(mid.max())
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    ax.set_ylim(ymin, ymax)

    xmin, xmax = int(ts.min()), int(ts.max())
    if xmin == xmax:
        xmin -= 1
        xmax += 1
    ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    out = root / f"plot_{PRODUCT}_r{ROUND}_day{day}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
