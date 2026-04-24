#!/usr/bin/env python3
"""For each day, list the 5 ASH_COATED_OSMIUM trades closest to that day's min and max trade price.

Olivia / Hedgehogs-style *direction* filter (optional, on by default): they looked for
extrema prints **in the expected direction vs mid** — buying at the low, selling at the
high (see Prosperity3Winner/3Writeup.txt, Squid Ink). With only price + mid (no trader
ID), we approximate aggressor side with a tick rule:

  * Near **daily min**: keep trades with **price >= mid** (buy-side / at-or-above mid).
  * Near **daily max**: keep trades with **price <= mid** (sell-side / at-or-below mid).

Mids come from ``prices_round_*`` (``mid_price``, nonzero), joined with
``pd.merge_asof(..., direction="backward")`` on ``timestamp``.

Use ``--include-wrong-side`` to list closest trades **without** this filter.

Also prints **every** trade whose price equals that day’s **trade min** or **trade max**
(exact hit on the daily low / high of executed prices).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRADES_DIR = REPO / "Prosperity4Data" / "ROUND1"
PRODUCT = "ASH_COATED_OSMIUM"
ROUND = 1
K = 5


def days() -> list[int]:
    out: list[int] = []
    for p in TRADES_DIR.glob(f"trades_round_{ROUND}_day_*.csv"):
        m = re.search(r"day_(-?\d+)\.csv$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load(day: int) -> pd.DataFrame:
    path = TRADES_DIR / f"trades_round_{ROUND}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["symbol"] == PRODUCT, ["timestamp", "price", "quantity", "buyer", "seller"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    return df.dropna(subset=["price"])


def load_mids(day: int) -> pd.DataFrame:
    path = TRADES_DIR / f"prices_round_{ROUND}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["product"] == PRODUCT, ["timestamp", "mid_price"]].copy()
    df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
    df = df.loc[(df["mid_price"].notna()) & (df["mid_price"] != 0)]
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df


def attach_mid(trades: pd.DataFrame, mids: pd.DataFrame) -> pd.DataFrame:
    if mids.empty:
        out = trades.copy()
        out["_mid"] = float("nan")
        return out
    t = trades.sort_values("timestamp").copy()
    m = mids.rename(columns={"mid_price": "_mid"})
    return pd.merge_asof(t, m, on="timestamp", direction="backward")


def _fmt_bs(r: pd.Series) -> tuple[str, str]:
    b = r.get("buyer")
    s = r.get("seller")
    b = "—" if pd.isna(b) or str(b).strip() == "" else str(b).strip()
    s = "—" if pd.isna(s) or str(s).strip() == "" else str(s).strip()
    return b, s


def show_extrema_hits(title: str, sub: pd.DataFrame) -> None:
    """All trades at exactly the day’s min or max print (no direction filter)."""
    print(title)
    if sub.empty:
        print("  (none)")
        return
    for _, r in sub.iterrows():
        b, s = _fmt_bs(r)
        mid_s = ""
        if "_mid" in r.index and pd.notna(r["_mid"]):
            mid_s = f"  mid={float(r['_mid']):.1f}  px−mid={float(r['price']) - float(r['_mid']):+.1f}"
        print(
            f"  ts={int(r['timestamp']):6d}  px={float(r['price']):.1f}  qty={int(r['quantity'])}"
            f"{mid_s}  buyer={b}  seller={s}"
        )


def show_block(title: str, sub: pd.DataFrame, *, show_mid: bool) -> None:
    print(title)
    if sub.empty:
        print("  (none)")
        return
    for _, r in sub.iterrows():
        b, s = _fmt_bs(r)
        mid_s = ""
        if show_mid and "_mid" in r.index and pd.notna(r["_mid"]):
            mid_s = f"  mid={float(r['_mid']):.1f}  px−mid={float(r['price']) - float(r['_mid']):+.1f}"
        print(
            f"  ts={int(r['timestamp']):6d}  px={float(r['price']):.1f}  qty={int(r['quantity'])}  "
            f"Δ={float(r['_dist']):.2f}{mid_s}  buyer={b}  seller={s}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Closest osmium trades to daily min/max trade price.")
    ap.add_argument(
        "--include-wrong-side",
        action="store_true",
        help="Do not filter by Olivia-style direction vs mid (show all closest prints).",
    )
    ap.add_argument(
        "--no-closest",
        action="store_true",
        help="Only print trades at exact daily min/max (skip the top-5 closest lists).",
    )
    args = ap.parse_args()
    direction_filter = not args.include_wrong_side

    for day in days():
        df = load(day)
        if df.empty:
            print(f"\n=== day {day} ===\n(no trades)\n")
            continue
        mids = load_mids(day)
        dfm = attach_mid(df, mids)
        n_no_mid = int(dfm["_mid"].isna().sum())

        pmin = float(df["price"].min())
        pmax = float(df["price"].max())
        dfm["_dist_min"] = (dfm["price"] - pmin).abs()
        dfm["_dist_max"] = (dfm["price"] - pmax).abs()

        if direction_filter:
            pool_min = dfm.loc[dfm["_mid"].notna() & (dfm["price"] >= dfm["_mid"])].copy()
            pool_max = dfm.loc[dfm["_mid"].notna() & (dfm["price"] <= dfm["_mid"])].copy()
            closest_tag = (
                "  [closest-5 uses Olivia-style filter: min side → price≥mid; "
                "max side → price≤mid]"
            )
        else:
            pool_min = dfm.copy()
            pool_max = dfm.copy()
            closest_tag = "  [closest-5: no direction filter]"

        near_min = pool_min.nsmallest(K, "_dist_min").copy()
        near_min["_dist"] = near_min["_dist_min"]
        near_max = pool_max.nsmallest(K, "_dist_max").copy()
        near_max["_dist"] = near_max["_dist_max"]

        at_min = dfm.loc[dfm["price"] == pmin].copy()
        at_max = dfm.loc[dfm["price"] == pmax].copy()

        head = (
            f"\n=== day {day} ===  trade min={pmin:.1f}  trade max={pmax:.1f}  n={len(df)}"
            f"  mid rows={len(mids)}  trades w/o mid join={n_no_mid}"
        )
        if not args.no_closest:
            head += closest_tag
        print(head)
        show_extrema_hits(
            f"**All trades at daily MIN** ({pmin:.1f}), n={len(at_min)}  (exact trade low):",
            at_min,
        )
        show_extrema_hits(
            f"**All trades at daily MAX** ({pmax:.1f}), n={len(at_max)}  (exact trade high):",
            at_max,
        )
        if not args.no_closest:
            show_block(
                f"5 closest to **daily min** ({pmin:.1f})"
                + (" among price≥mid" if direction_filter else "")
                + ":",
                near_min,
                show_mid=direction_filter,
            )
            show_block(
                f"5 closest to **daily max** ({pmax:.1f})"
                + (" among price≤mid" if direction_filter else "")
                + ":",
                near_max,
                show_mid=direction_filter,
            )
        print()


if __name__ == "__main__":
    main()
