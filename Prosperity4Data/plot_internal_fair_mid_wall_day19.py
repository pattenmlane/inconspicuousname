#!/usr/bin/env python3
"""
Plot **true internal fair** ``F = E + PnL`` vs several book mids for
ASH_COATED_OSMIUM (Round 1 day 19 = submission tape):

  * **touch mid** — best bid + best ask (only if ≥1 bid & ≥1 ask with vol > 0)
  * **wall mid** — min bid + max ask over those levels
  * **popular mid** — price at max bid size + price at max ask size
  * **worst mid** — price at **min** bid size + price at **min** ask size (thin-liquidity anchor)

No ``mid_price`` CSV fallback — missing side → **NaN** so lines **gap** instead of spiking.

**Interactive** (omit ``--no-show``): use the matplotlib **toolbar** (zoom box, pan,
home). **Checkboxes** on the right toggle fair curves and **market-trade** overlays.
Requires a GUI backend (TkAgg/QtAgg on Mac); SSH/headless → use ``--no-show`` only.

**Market trades** (``--trades-csv``): public tape rows for osmium from the **trades**
CSV (not the price tape). ``SUBMISSION`` fills from ``--pnl-log`` ``tradeHistory``
are dropped so you see **others'** prints only. Aggressor side is inferred from the
**price book** at the same ``timestamp`` (touch bid / touch ask from ``--day-csv``):
``price >= best_ask`` → **taker buy**, ``price <= best_bid`` → **taker sell**,
otherwise **inside spread**. Marker **size** encodes quantity bucket: 1 lot, 2–5,
6+.

Usage::

  python3 Prosperity4Data/plot_internal_fair_mid_wall_day19.py \\
    --pnl-log INK_INFO/248329.log --entry-from-log INK_INFO/248329.log

  python3 Prosperity4Data/plot_internal_fair_mid_wall_day19.py \\
    --entry-price 10011 --pnl-log INK_INFO/248329.log --tmin 0 --tmax 200000

  python3 Prosperity4Data/plot_internal_fair_mid_wall_day19.py ... --no-show
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons

ROUND = 1
DAY = 19
PRODUCT = "ASH_COATED_OSMIUM"


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
    """Max-volume bid + max-volume ask (same as vol mid in other scripts)."""
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    pop_b = max(b, key=lambda t: t[1])[0]
    pop_a = max(a, key=lambda t: t[1])[0]
    return (pop_b + pop_a) / 2.0


def worst_mid_row(row: pd.Series) -> float:
    """Thinnest-size bid + thinnest ask (requires ≥1 bid & ≥1 ask with vol > 0)."""
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return float("nan")
    thin_b = min(b, key=lambda t: t[1])[0]
    thin_a = min(a, key=lambda t: t[1])[0]
    return (thin_b + thin_a) / 2.0


def entry_price_from_log(log_path: Path) -> float:
    obj = json.loads(log_path.read_text(encoding="utf-8"))
    th = obj.get("tradeHistory")
    if not isinstance(th, list):
        raise SystemExit("log JSON missing tradeHistory")
    for tr in th:
        if tr.get("buyer") != "SUBMISSION" or tr.get("symbol") != PRODUCT:
            continue
        if int(tr.get("quantity", 0)) <= 0:
            continue
        return float(tr["price"])
    raise SystemExit(f"No SUBMISSION buy for {PRODUCT} in {log_path}")


def load_pnl_series(log_path: Path) -> pd.Series:
    obj = json.loads(log_path.read_text(encoding="utf-8"))
    al = obj.get("activitiesLog")
    if not isinstance(al, str):
        raise SystemExit("log JSON missing activitiesLog")
    dlog = pd.read_csv(io.StringIO(al), sep=";")
    return dlog.set_index(["timestamp", "product"])["profit_and_loss"]


def submission_trade_keys(log_path: Path) -> set[tuple[int, float, int]]:
    """(timestamp, price, quantity) for SUBMISSION-anchored prints on this product."""
    obj = json.loads(log_path.read_text(encoding="utf-8"))
    th = obj.get("tradeHistory")
    if not isinstance(th, list):
        return set()
    keys: set[tuple[int, float, int]] = set()
    for tr in th:
        if tr.get("symbol") != PRODUCT:
            continue
        if tr.get("buyer") != "SUBMISSION" and tr.get("seller") != "SUBMISSION":
            continue
        q = int(tr.get("quantity", 0))
        if q <= 0:
            continue
        keys.add((int(tr["timestamp"]), float(tr["price"]), q))
    return keys


def best_bid_ask(row: pd.Series) -> tuple[float | None, float | None]:
    b, a = _levels(row, "bid"), _levels(row, "ask")
    if not b or not a:
        return None, None
    return max(p for p, _ in b), min(p for p, _ in a)


def classify_aggressor(price: float, row: pd.Series) -> str:
    """
    Inferred aggressor (taker side). Resting counterparty is the *maker*.
    Returns: taker_buy | taker_sell | inside | no_book
    """
    bb, ba = best_bid_ask(row)
    if bb is None or ba is None:
        return "no_book"
    if price >= ba:
        return "taker_buy"
    if price <= bb:
        return "taker_sell"
    return "inside"


def qty_bucket(q: int) -> int:
    if q <= 1:
        return 0
    if q <= 5:
        return 1
    return 2


def scatter_pt_size(bucket: int) -> float:
    return (28.0, 55.0, 95.0)[bucket]


def load_market_trades_classified(
    trades_csv: Path,
    prices_df: pd.DataFrame,
    submission_keys: set[tuple[int, float, int]],
    tmin: int | None,
    tmax: int | None,
) -> pd.DataFrame:
    if not trades_csv.is_file():
        raise SystemExit(f"Missing trades CSV: {trades_csv}")
    raw = pd.read_csv(trades_csv, sep=";")
    tr = raw.loc[raw["symbol"] == PRODUCT].copy()
    if tr.empty:
        return pd.DataFrame()
    tr["timestamp"] = tr["timestamp"].astype(int)
    tr["price"] = tr["price"].astype(float)
    tr["quantity"] = tr["quantity"].astype(int)
    keys = list(
        zip(tr["timestamp"].to_numpy(), tr["price"].to_numpy(), tr["quantity"].to_numpy())
    )
    drop = {i for i, k in enumerate(keys) if k in submission_keys}
    tr = tr.drop(index=list(drop)).reset_index(drop=True)
    if tmin is not None:
        tr = tr.loc[tr["timestamp"] >= tmin]
    if tmax is not None:
        tr = tr.loc[tr["timestamp"] <= tmax]
    if tr.empty:
        return tr

    osm = prices_df.loc[prices_df["product"] == PRODUCT, :].drop_duplicates(subset=["timestamp"])
    osm = osm.set_index("timestamp", drop=False)

    aggr: list[str] = []
    for _, r in tr.iterrows():
        ts = int(r["timestamp"])
        if ts not in osm.index:
            aggr.append("no_book")
            continue
        aggr.append(classify_aggressor(float(r["price"]), osm.loc[ts]))
    tr["aggressor"] = aggr
    tr["qty_bucket"] = tr["quantity"].map(qty_bucket)
    tr["pt_size"] = tr["qty_bucket"].map(scatter_pt_size)

    # Horizontal jitter when several trades share one timestamp
    tr = tr.sort_values(["timestamp", "price"]).reset_index(drop=True)
    gsz = tr.groupby("timestamp").cumcount()
    gn = tr.groupby("timestamp")["timestamp"].transform("size")
    tr["_xplot"] = tr["timestamp"].to_numpy() + (gsz.to_numpy() - (gn.to_numpy() - 1) / 2.0) * 18.0
    return tr


def main() -> None:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--day-csv",
        type=Path,
        default=root / "ROUND1" / f"prices_round_{ROUND}_day_{DAY}.csv",
    )
    ap.add_argument("--pnl-log", type=Path, required=True)
    ap.add_argument("--entry-price", type=float, default=None)
    ap.add_argument("--entry-from-log", type=Path, default=None)
    ap.add_argument("--tmin", type=int, default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Save PNG only, do not open interactive window.",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: ROUND1/plot_internal_fair_mid_wall_r1_day19.png)",
    )
    ap.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
        help="Market trades file (default: ROUND1/trades_round_<r>_day_<d>.csv next to day-csv).",
    )
    args = ap.parse_args()

    if not args.day_csv.is_file():
        sys.exit(f"Missing {args.day_csv}")
    if not args.pnl_log.is_file():
        sys.exit(f"Missing {args.pnl_log}")

    if args.entry_from_log:
        entry = entry_price_from_log(args.entry_from_log)
    elif args.entry_price is not None:
        entry = float(args.entry_price)
    else:
        sys.exit("Provide --entry-price or --entry-from-log")

    df = pd.read_csv(args.day_csv, sep=";")
    ser = load_pnl_series(args.pnl_log)

    def pnl_at(r: pd.Series) -> float:
        k = (int(r["timestamp"]), str(r["product"]))
        return float(ser.loc[k]) if k in ser.index else float("nan")

    sub = df.loc[df["product"] == PRODUCT].sort_values("timestamp").copy()
    sub["pnl"] = sub.apply(pnl_at, axis=1)
    sub["internal_fair"] = entry + sub["pnl"]
    sub["wall_mid"] = sub.apply(wall_mid_row, axis=1)
    # Book mids only when _levels finds ≥1 bid and ≥1 ask (vol > 0). NaNs → line gaps.
    sub["touch_mid"] = sub.apply(micro_mid_row, axis=1)
    sub["popular_mid"] = sub.apply(popular_mid_row, axis=1)
    sub["worst_mid"] = sub.apply(worst_mid_row, axis=1)

    if args.tmin is not None:
        sub = sub.loc[sub["timestamp"] >= args.tmin]
    if args.tmax is not None:
        sub = sub.loc[sub["timestamp"] <= args.tmax]
    if sub.empty:
        sys.exit("No rows after filters.")

    ts = sub["timestamp"].to_numpy()

    trades_csv = args.trades_csv
    if trades_csv is None:
        trades_csv = root / "ROUND1" / f"trades_round_{ROUND}_day_{DAY}.csv"
    sub_keys = submission_trade_keys(args.pnl_log)
    trades_enr = load_market_trades_classified(
        trades_csv,
        df,
        sub_keys,
        args.tmin,
        args.tmax,
    )

    fig = plt.figure(figsize=(16, 6.5))
    ax = fig.add_axes((0.06, 0.14, 0.58, 0.78))
    rax_lines = fig.add_axes((0.66, 0.52, 0.20, 0.38))
    rax_lines.set_title("Series", fontsize=9, pad=4)
    rax_trades = fig.add_axes((0.66, 0.08, 0.20, 0.38))
    rax_trades.set_title("Market trades", fontsize=9, pad=4)

    line_specs: list[tuple[str, np.ndarray, dict]] = [
        ("Internal fair", sub["internal_fair"].to_numpy(), {"color": "#1a6b2e", "lw": 1.5, "zorder": 5}),
        ("Touch mid", sub["touch_mid"].to_numpy(), {"color": "#3a3a8c", "lw": 1.1, "alpha": 0.9}),
        ("Wall mid", sub["wall_mid"].to_numpy(), {"color": "#b35900", "lw": 1.15, "alpha": 0.95}),
        ("Popular mid", sub["popular_mid"].to_numpy(), {"color": "#6b2d9e", "lw": 1.05, "alpha": 0.95}),
        ("Worst mid", sub["worst_mid"].to_numpy(), {"color": "#8b4513", "lw": 1.05, "alpha": 0.9, "ls": "--"}),
    ]

    lines: dict[str, Line2D] = {}
    for label, y, kw in line_specs:
        (ln,) = ax.plot(ts, y, **kw)
        lines[label] = ln

    ax.set_title(f"{PRODUCT} — R{ROUND} day {DAY}: internal fair vs book mids + market trades")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.35)
    ax.margins(x=0.01)

    ycols = ["internal_fair", "touch_mid", "wall_mid", "popular_mid", "worst_mid"]
    yvals = sub[ycols].to_numpy(dtype=float).ravel()
    yvals = yvals[np.isfinite(yvals)]
    ymin = float(yvals.min()) if len(yvals) else 0.0
    ymax = float(yvals.max()) if len(yvals) else 1.0
    pad = max(0.5, (ymax - ymin) * 0.04)
    ax.set_ylim(ymin - pad, ymax + pad)

    line_labels = tuple(lines.keys())
    line_check = CheckButtons(rax_lines, line_labels, (True,) * len(line_labels))

    def on_line_clicked(label: str) -> None:
        ln = lines[label]
        ln.set_visible(not ln.get_visible())
        fig.canvas.draw_idle()

    line_check.on_clicked(on_line_clicked)

    trade_collections: dict[str, PathCollection] = {}
    trade_style: dict[str, tuple[str, str]] = {
        "taker_buy": ("#0a6b38", "^"),
        "taker_sell": ("#b8182a", "v"),
        "inside": ("#444444", "o"),
        "no_book": ("#6a2c91", "D"),
    }
    trade_label: dict[str, str] = {
        "taker_buy": "Taker buy (hits ask)",
        "taker_sell": "Taker sell (hits bid)",
        "inside": "Inside spread",
        "no_book": "No touch book",
    }
    if not trades_enr.empty:
        for key, (color, marker) in trade_style.items():
            sl = trades_enr.loc[trades_enr["aggressor"] == key]
            if sl.empty:
                continue
            sc = ax.scatter(
                sl["_xplot"].to_numpy(),
                sl["price"].to_numpy(),
                s=sl["pt_size"].to_numpy(),
                c=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.35,
                alpha=0.88,
                zorder=6,
                label=trade_label[key],
            )
            trade_collections[trade_label[key]] = sc

    trade_labels = tuple(trade_collections.keys())
    if trade_labels:
        trade_check = CheckButtons(rax_trades, trade_labels, (True,) * len(trade_labels))

        def on_trade_clicked(label: str) -> None:
            pc = trade_collections[label]
            pc.set_visible(not pc.get_visible())
            fig.canvas.draw_idle()

        trade_check.on_clicked(on_trade_clicked)
    else:
        rax_trades.text(0.05, 0.5, "(no trades)", fontsize=8, transform=rax_trades.transAxes)

    fig.text(
        0.06,
        0.03,
        "Market prints: taker buy = price ≥ best ask; taker sell = price ≤ best bid; "
        "inside = strictly between touch. Marker size: small = 1 lot, medium = 2–5, large = 6+.",
        fontsize=7.5,
        va="bottom",
    )

    out = args.output or (root / "ROUND1" / f"plot_internal_fair_mid_wall_r{ROUND}_day{DAY}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if not trades_enr.empty:
        ct = trades_enr.groupby(["aggressor", "qty_bucket"]).size().unstack(fill_value=0)
        for c in (0, 1, 2):
            if c not in ct.columns:
                ct[c] = 0
        ct = ct[[0, 1, 2]].rename(columns={0: "1 lot", 1: "qty 2–5", 2: "qty 6+"})
        print("Market trades (SUBMISSION fills from log removed): count by aggressor × size bucket")
        print(ct.to_string())

    if args.no_show:
        plt.close(fig)
    else:
        print("Toolbar: zoom / pan / home. Checkboxes: toggle series and trade layers.")
        plt.show()


if __name__ == "__main__":
    main()
