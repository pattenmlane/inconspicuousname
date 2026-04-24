#!/usr/bin/env python3
"""
Plot ASH_COATED_OSMIUM from Round 1 price CSVs.

Modes:
  * Default: **micro mid** vs **popular (vol) mid** + difference panel.
  * ``--popular-only``: only popular mid.
  * ``--six-lines``: **best ask**, **best bid**, **pop ask**, **pop bid**, **best mid**, **pop mid**
    (best = touch from L2; pop = price at max displayed size per side; mids are averages).

``--tmin`` / ``--tmax`` restrict the timestamp window (inclusive).

**Interactive zoom:** omit ``--no-show`` so a GUI window opens (matplotlib’s toolbar:
zoom box, pan, home/reset). Same if you pass ``--interactive`` / ``-i`` (overrides
``--no-show`` if both are given). Run from a normal terminal on your Mac, not a
headless/SSH session, and use a GUI matplotlib backend (default is usually fine).

Usage:
  python3 Prosperity4Data/plot_osmium_micro_mid_vs_vol_mid.py --day -2 --six-lines --interactive
  python3 Prosperity4Data/plot_osmium_micro_mid_vs_vol_mid.py -d -2 --tmin 400000 --tmax 500000 --no-show
  python3 Prosperity4Data/plot_osmium_micro_mid_vs_vol_mid.py --day -2 --popular-only --no-show
  python3 Prosperity4Data/plot_osmium_micro_mid_vs_vol_mid.py --day -2 --six-lines --tmin 600000 --tmax 800000 --no-show
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


def _levels(row: pd.Series, side: str) -> list[tuple[float, float]]:
    """List of (price, volume) for side in ('bid','ask') with vol > 0 and price present."""
    out: list[tuple[float, float]] = []
    for i in range(1, 4):
        p = row.get(f"{side}_price_{i}")
        v = row.get(f"{side}_volume_{i}")
        if pd.isna(p) or pd.isna(v):
            continue
        v = float(v)
        if v <= 0:
            continue
        out.append((float(p), v))
    return out


def micro_mid_row(row: pd.Series) -> float | None:
    bids = _levels(row, "bid")
    asks = _levels(row, "ask")
    if not bids or not asks:
        return None
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    return (best_bid + best_ask) / 2.0


def vol_mid_row(row: pd.Series) -> float | None:
    bids = _levels(row, "bid")
    asks = _levels(row, "ask")
    if not bids or not asks:
        return None
    popular_bid = max(bids, key=lambda t: t[1])[0]
    popular_ask = max(asks, key=lambda t: t[1])[0]
    return (popular_bid + popular_ask) / 2.0


def six_series_row(row: pd.Series) -> pd.Series:
    """One row → best/pop prices and mids; NaNs if book incomplete."""
    nan6 = pd.Series(
        {
            "best_ask": float("nan"),
            "best_bid": float("nan"),
            "pop_ask": float("nan"),
            "pop_bid": float("nan"),
            "best_mid": float("nan"),
            "pop_mid": float("nan"),
        }
    )
    bids = _levels(row, "bid")
    asks = _levels(row, "ask")
    if not bids or not asks:
        return nan6
    best_bid = max(p for p, _ in bids)
    best_ask = min(p for p, _ in asks)
    pop_bid = max(bids, key=lambda t: t[1])[0]
    pop_ask = max(asks, key=lambda t: t[1])[0]
    return pd.Series(
        {
            "best_ask": best_ask,
            "best_bid": best_bid,
            "pop_ask": pop_ask,
            "pop_bid": pop_bid,
            "best_mid": (best_bid + best_ask) / 2.0,
            "pop_mid": (pop_bid + pop_ask) / 2.0,
        }
    )


def load_raw(root: Path, day: int) -> pd.DataFrame:
    path = root / f"prices_round_{ROUND}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["product"] == PRODUCT].copy()
    return df.sort_values("timestamp")


def load_day(root: Path, day: int) -> pd.DataFrame:
    df = load_raw(root, day)
    df["micro_mid"] = df.apply(micro_mid_row, axis=1)
    df["vol_mid"] = df.apply(vol_mid_row, axis=1)
    return df.loc[df["micro_mid"].notna() & df["vol_mid"].notna()].copy()


def load_day_six(root: Path, day: int) -> pd.DataFrame:
    df = load_raw(root, day)
    extra = df.apply(six_series_row, axis=1)
    df = pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    cols = ["best_ask", "best_bid", "pop_ask", "pop_bid", "best_mid", "pop_mid"]
    return df.dropna(subset=cols, how="any").copy()


def main() -> None:
    p = argparse.ArgumentParser(description=f"Plot {PRODUCT} book-derived series.")
    p.add_argument("--day", "-d", type=int, default=-2, help="Trading day (default -2).")
    p.add_argument("--tmin", type=int, default=None, help="Only timestamps >= this (inclusive).")
    p.add_argument("--tmax", type=int, default=None, help="Only timestamps <= this (inclusive).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--popular-only",
        action="store_true",
        help="Plot only popular mid (single line).",
    )
    mode.add_argument(
        "--six-lines",
        action="store_true",
        help="Plot best ask/bid, pop ask/bid, best mid, pop mid (six lines).",
    )
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Open a zoomable matplotlib window after saving (overrides --no-show).",
    )
    p.add_argument("--no-show", action="store_true", help="Save PNG only; do not open a window.")
    args = p.parse_args()
    if args.interactive and args.no_show:
        print("Note: --interactive overrides --no-show (GUI will open).")
    day = args.day

    root = _data_dir()
    if not root.is_dir():
        raise SystemExit(f"Missing data folder: {root}")

    if args.six_lines:
        df = load_day_six(root, day)
        suffix = "_six_lines"
    else:
        df = load_day(root, day)
        suffix = "_popular_mid" if args.popular_only else "_micro_vs_volmid"

    if df.empty:
        raise SystemExit(f"No plottable rows for {PRODUCT} on day {day}.")

    if args.tmin is not None:
        df = df.loc[df["timestamp"] >= args.tmin].copy()
    if args.tmax is not None:
        df = df.loc[df["timestamp"] <= args.tmax].copy()
    if df.empty:
        raise SystemExit(
            f"No rows left after timestamp filter [{args.tmin}, {args.tmax}] for day {day}."
        )

    ts = df["timestamp"]
    ttag = ""
    if args.tmin is not None or args.tmax is not None:
        ttag = f" (timestamp {args.tmin}–{args.tmax})"

    xmin = int(args.tmin) if args.tmin is not None else int(ts.min())
    xmax = int(args.tmax) if args.tmax is not None else int(ts.max())
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if xmin == xmax:
        xmin -= 1
        xmax += 1

    if args.six_lines:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        styles = [
            ("best_ask", "best ask", "#c53030", "-"),
            ("best_bid", "best bid", "#2f855a", "-"),
            ("pop_ask", "pop ask (max vol ask px)", "#f56565", "--"),
            ("pop_bid", "pop bid (max vol bid px)", "#48bb78", "--"),
            ("best_mid", "best mid", "#2c5282", "-"),
            ("pop_mid", "pop mid", "#805ad5", "-"),
        ]
        ylo, yhi = float("inf"), float("-inf")
        for col, lab, c, ls in styles:
            y = df[col].astype(float)
            ax.plot(ts, y, linewidth=0.85, color=c, linestyle=ls, label=lab)
            ylo = min(ylo, float(y.min()))
            yhi = max(yhi, float(y.max()))
        if ylo == yhi or ylo == float("inf"):
            ylo, yhi = ylo - 1.0, yhi + 1.0
        else:
            pad = max(1.0, (yhi - ylo) * 0.02)
            ylo -= pad
            yhi += pad
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(xmin, xmax)
        ax.set_title(f"{PRODUCT} — Round {ROUND}, day {day}: six series{ttag}")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("price")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.35)
    elif args.popular_only:
        m_vol = df["vol_mid"]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ts, m_vol, linewidth=0.9, color="#c05621", label="Popular mid (max-vol bid + max-vol ask) / 2")
        ax.set_title(f"{PRODUCT} — Round {ROUND}, day {day}: popular mid{ttag}")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("popular mid")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.35)
        ylo, yhi = float(m_vol.min()), float(m_vol.max())
        if ylo == yhi:
            ylo -= 1.0
            yhi += 1.0
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(xmin, xmax)
    else:
        m_micro = df["micro_mid"]
        m_vol = df["vol_mid"]
        diff = m_vol - m_micro
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
        ax0 = axes[0]
        ax0.plot(ts, m_micro, linewidth=0.85, color="#2c5282", label="Micro mid (best bid + best ask) / 2")
        ax0.plot(ts, m_vol, linewidth=0.85, color="#c05621", alpha=0.9, label="Vol mid (max-vol bid + max-vol ask) / 2")
        ax0.set_title(f"{PRODUCT} — Round {ROUND}, day {day}: micro mid vs volume mid{ttag}")
        ax0.set_ylabel("price")
        ax0.legend(loc="upper right", fontsize=9)
        ax0.grid(True, alpha=0.35)
        ylo = min(float(m_micro.min()), float(m_vol.min()))
        yhi = max(float(m_micro.max()), float(m_vol.max()))
        if ylo == yhi:
            ylo -= 1.0
            yhi += 1.0
        ax0.set_ylim(ylo, yhi)
        ax1 = axes[1]
        ax1.plot(ts, diff, linewidth=0.7, color="#553c7b")
        ax1.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
        ax1.set_ylabel("vol_mid − micro_mid")
        ax1.set_xlabel("timestamp")
        ax1.grid(True, alpha=0.35)
        ax1.set_xlim(xmin, xmax)

    plt.tight_layout()
    safe = PRODUCT.replace(" ", "_")
    if args.tmin is not None or args.tmax is not None:
        suffix += f"_t{args.tmin}-{args.tmax}"
    out = root / f"plot_{safe}_r{ROUND}_day{day}{suffix}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    show_gui = args.interactive or not args.no_show
    if show_gui:
        print("Opening figure: toolbar → zoom (rectangle), pan (hand), home (reset). Close window to exit.")
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
