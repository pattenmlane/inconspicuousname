#!/usr/bin/env python3
"""
Plot ASH_COATED_OSMIUM mid (and optional bid/ask) over a timestamp window.

Usage:
  python3 Prosperity4Data/plot_osmium_window.py --day -2 --tmin 40000 --tmax 50000
  python3 Prosperity4Data/plot_osmium_window.py -d -2 --tmin 40000 --tmax 50000 --no-show
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


def main() -> None:
    p = argparse.ArgumentParser(description=f"Plot {PRODUCT} over a timestamp window.")
    p.add_argument("--day", "-d", type=int, default=-2)
    p.add_argument("--tmin", type=int, required=True)
    p.add_argument("--tmax", type=int, required=True)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    path = _data_dir() / f"prices_round_{ROUND}_day_{args.day}.csv"
    if not path.is_file():
        raise SystemExit(f"Missing: {path}")

    df = pd.read_csv(path, sep=";")
    df = df[df["product"] == PRODUCT].sort_values("timestamp")
    df = df[(df["timestamp"] >= args.tmin) & (df["timestamp"] <= args.tmax)].copy()

    if df.empty:
        raise SystemExit(f"No rows in [{args.tmin}, {args.tmax}] for {PRODUCT} day {args.day}.")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["mid_price"], color="#4a4a8c", lw=1.0, label="mid_price")
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        ax.plot(df["timestamp"], df["bid_price_1"], color="#2a7a3a", lw=0.6, alpha=0.7, label="bid1")
        ax.plot(df["timestamp"], df["ask_price_1"], color="#8c3a3a", lw=0.6, alpha=0.7, label="ask1")
    ax.axhline(10000, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_title(f"{PRODUCT} — R{ROUND} day {args.day}, t ∈ [{args.tmin}, {args.tmax}]")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("price")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.35)
    ymin = float(df["mid_price"].replace(0, pd.NA).dropna().min())
    ymax = float(df["mid_price"].replace(0, pd.NA).dropna().max())
    if ymin == ymax:
        ymin -= 1
        ymax += 1
    pad = max(0.5, (ymax - ymin) * 0.05)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(args.tmin, args.tmax)
    plt.tight_layout()

    out = _data_dir() / f"plot_{PRODUCT}_r{ROUND}_day{args.day}_t{args.tmin}-{args.tmax}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
