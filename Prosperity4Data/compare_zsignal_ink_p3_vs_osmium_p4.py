#!/usr/bin/env python3
"""
Compare **jmerle-style smoothed rolling z** signal strength:

  * **Prosperity 3 Round 1** — ``SQUID_INK`` (last year), ``Prosperity3Data/round1/``
  * **Prosperity 4 Round 1** — ``ASH_COATED_OSMIUM`` (this year), ``Prosperity4Data/ROUND1/``

Same mid definition for both: ``--mid jmerle`` (max-vol bid + min-vol ask) by default.

**Strength metrics** (higher = stronger mean-reversion *signal* in the log; not PnL):
  * ``neg_corr_f1`` = ``-corr(signal, f1)`` — want positive if high z predicts next tick down.
  * ``sym_f1`` = ``|mean(f1|sig>thr)| + |mean(f1|sig<-thr)|`` — size of conditional next-step move.

Then **grid-search** ``(Wz, Ws)`` on osmium only to best match ink’s ``neg_corr_f1`` at a
reference window (default **50, 30**, same spirit as exploratory short windows).

Usage:
  python3 Prosperity4Data/compare_zsignal_ink_p3_vs_osmium_p4.py
  python3 Prosperity4Data/compare_zsignal_ink_p3_vs_osmium_p4.py --ref-wz 30 --ref-ws 20 --grid-max-wz 120
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_osmium_wall_mid_spikes import wall_mid_row  # noqa: E402
from plot_osmium_micro_mid_vs_vol_mid import (  # noqa: E402
    _levels,
    micro_mid_row,
    vol_mid_row,
)


def load_raw_product(root: Path, day: int, product: str, round_n: int = 1) -> pd.DataFrame:
    path = root / f"prices_round_{round_n}_day_{day}.csv"
    df = pd.read_csv(path, sep=";")
    df = df.loc[df["product"] == product].copy()
    return df.sort_values("timestamp")


def discover_days(root: Path, round_n: int = 1) -> list[int]:
    days: list[int] = []
    for p in root.glob(f"prices_round_{round_n}_day_*.csv"):
        m = re.search(r"day_(-?\d+)\.csv$", p.name)
        if m:
            days.append(int(m.group(1)))
    return sorted(days)


def jmerle_pop_mid_row(row: pd.Series) -> float | None:
    bids = _levels(row, "bid")
    asks = _levels(row, "ask")
    if not bids or not asks:
        return None
    popular_buy = max(bids, key=lambda t: t[1])[0]
    popular_sell = min(asks, key=lambda t: t[1])[0]
    return (popular_buy + popular_sell) / 2.0


def mid_series(root: Path, days: list[int], product: str, mid_kind: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in days:
        df = load_raw_product(root, d, product)
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
    return pd.concat(frames, ignore_index=True)


def smoothed_z(mid: pd.Series, wz: int, ws: int) -> pd.Series:
    rmean = mid.rolling(wz, min_periods=wz).mean()
    rstd = mid.rolling(wz, min_periods=wz).std()
    z = (mid - rmean) / rstd
    return z.rolling(ws, min_periods=ws).mean()


def metrics_for_window(df: pd.DataFrame, wz: int, ws: int, thresh: float) -> dict[str, float] | None:
    df = df.copy()
    df["sig"] = df.groupby("day", sort=False)["m"].transform(lambda s: smoothed_z(s, wz, ws))
    df["f1"] = df.groupby("day", sort=False)["m"].transform(lambda s: s.shift(-1) - s)
    sub = df.loc[df["sig"].notna() & df["f1"].notna()]
    if len(sub) < 200:
        return None
    c = float(sub["sig"].corr(sub["f1"]))
    if np.isnan(c):
        return None
    hi = sub["sig"] > thresh
    lo = sub["sig"] < -thresh
    if hi.sum() < 20 or lo.sum() < 20:
        return None
    m_hi = float(sub.loc[hi, "f1"].mean())
    m_lo = float(sub.loc[lo, "f1"].mean())
    return {
        "n": float(len(sub)),
        "corr_f1": c,
        "mr_score": -c,
        "sym_f1": abs(m_hi) + abs(m_lo),
        "mean_f1_hi": m_hi,
        "mean_f1_lo": m_lo,
    }


def print_block(name: str, df: pd.DataFrame, windows: list[tuple[int, int]], thresh: float) -> None:
    print(f"=== {name} (rows={len(df)}) ===")
    for wz, ws in windows:
        m = metrics_for_window(df, wz, ws, thresh)
        if m is None:
            print(f"  Wz={wz} Ws={ws}: insufficient data")
            continue
        print(
            f"  Wz={wz:3d} Ws={ws:3d}  corr(sig,f1)={m['corr_f1']:+.4f}  MR_score=-corr={m['mr_score']:+.4f}  "
            f"sym_f1={m['sym_f1']:.4f}  (mean f1|sig>thr={m['mean_f1_hi']:+.4f}, "
            f"mean f1|sig<-thr={m['mean_f1_lo']:+.4f})"
        )
    print()


def grid_match_mr_score(
    df_osm: pd.DataFrame,
    target_mr: float,
    thresh: float,
    wz_min: int,
    wz_max: int,
    ws_min: int,
) -> list[tuple[float, int, int]]:
    """Match MR_score = -corr(sig,f1). Smaller squared error first."""
    scored: list[tuple[float, int, int]] = []
    for wz in range(wz_min, wz_max + 1, 2):
        for ws in range(ws_min, wz, 2):
            m = metrics_for_window(df_osm, wz, ws, thresh)
            if m is None:
                continue
            err = (m["mr_score"] - target_mr) ** 2
            scored.append((err, wz, ws))
    scored.sort(key=lambda t: t[0])
    return scored


def grid_match_sym(
    df_osm: pd.DataFrame,
    target_sym: float,
    thresh: float,
    wz_min: int,
    wz_max: int,
    ws_min: int,
    min_mr: float,
) -> list[tuple[float, int, int]]:
    """Match sym_f1; require MR_score >= min_mr (fade-consistent negative corr)."""
    scored: list[tuple[float, int, int]] = []
    for wz in range(wz_min, wz_max + 1, 2):
        for ws in range(ws_min, wz, 2):
            m = metrics_for_window(df_osm, wz, ws, thresh)
            if m is None or m["mr_score"] < min_mr:
                continue
            err = (m["sym_f1"] - target_sym) ** 2
            scored.append((err, wz, ws))
    scored.sort(key=lambda t: t[0])
    return scored


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mid", default="jmerle", choices=("jmerle", "vol", "wall", "micro", "csv_mid"))
    p.add_argument("--thresh", type=float, default=1.0)
    p.add_argument(
        "--ref-wz",
        type=int,
        default=150,
        help="Ink reference Wz (default 150 = jmerle trader).",
    )
    p.add_argument("--ref-ws", type=int, default=100)
    p.add_argument("--grid-max-wz", type=int, default=150)
    p.add_argument("--grid-top", type=int, default=12, help="How many closest (Wz,Ws) pairs to print.")
    p.add_argument(
        "--osmium-extra-days",
        action="store_true",
        help="Include all osmium CSV days; default is only days that exist for ink too (-2,-1,0).",
    )
    args = p.parse_args()

    ink_root = _REPO / "Prosperity3Data" / "round1"
    osm_root = _REPO / "Prosperity4Data" / "ROUND1"
    ink_product = "SQUID_INK"
    osm_product = "ASH_COATED_OSMIUM"

    ink_days = discover_days(ink_root)
    osm_all = discover_days(osm_root)
    if not ink_days:
        raise SystemExit(f"No ink CSVs under {ink_root}")
    if not osm_all:
        raise SystemExit(f"No osmium CSVs under {osm_root}")
    osm_days = osm_all if args.osmium_extra_days else [d for d in osm_all if d in ink_days]
    if not osm_days:
        raise SystemExit("No overlapping days between ink and osmium.")

    df_ink = mid_series(ink_root, ink_days, ink_product, args.mid)
    df_osm = mid_series(osm_root, osm_days, osm_product, args.mid)

    windows = [(20, 15), (30, 20), (50, 30), (80, 50), (150, 100)]
    print("Jmerle-style smoothed z — signal strength comparison")
    print(f"mid={args.mid!r}  threshold=±{args.thresh}")
    print(f"Ink (P3 R1):  {ink_root}  days={ink_days}  n={len(df_ink)}")
    print(
        f"Osm (P4 R1):  {osm_root}  days={osm_days}  n={len(df_osm)}"
        + ("  [extra days]" if args.osmium_extra_days else "  [ink day overlap only]")
    )
    print()
    print_block("SQUID_INK (Prosperity 3)", df_ink, windows, args.thresh)
    print_block("ASH_COATED_OSMIUM (Prosperity 4)", df_osm, windows, args.thresh)

    ref_m = metrics_for_window(df_ink, args.ref_wz, args.ref_ws, args.thresh)
    if ref_m is None:
        raise SystemExit("Reference window on ink failed.")
    target_mr = ref_m["mr_score"]
    target_sym = ref_m["sym_f1"]
    osm_same = metrics_for_window(df_osm, args.ref_wz, args.ref_ws, args.thresh)
    if osm_same is None:
        raise SystemExit("Same window on osmium failed.")
    print(
        f"Ink reference: Wz={args.ref_wz} Ws={args.ref_ws}  →  corr(sig,f1)={ref_m['corr_f1']:+.4f}  "
        f"MR_score=-corr={target_mr:+.4f}  sym_f1={target_sym:.4f}"
    )
    print(
        f"Osmium same window: corr={osm_same['corr_f1']:+.4f}  MR_score={osm_same['mr_score']:+.4f}  sym_f1={osm_same['sym_f1']:.4f}"
    )
    print()
    print(
        f"Grid A — match MR_score ≈ ink ({target_mr:+.4f}), Wz=8..{args.grid_max_wz} step 2 …"
    )
    ranked = grid_match_mr_score(df_osm, target_mr, args.thresh, wz_min=8, wz_max=args.grid_max_wz, ws_min=2)
    if not ranked:
        raise SystemExit("Grid A produced no valid osmium windows.")
    print(f"Top {args.grid_top} by squared error on MR_score:")
    for i, (err, wz, ws) in enumerate(ranked[: args.grid_top], 1):
        m = metrics_for_window(df_osm, wz, ws, args.thresh)
        assert m is not None
        print(
            f"  {i:2d}. Wz={wz:3d} Ws={ws:3d}  MR_score={m['mr_score']:+.4f}  sym_f1={m['sym_f1']:.4f}  err²={err:.2e}"
        )
    print()
    print(
        f"Grid B — match sym_f1 ≈ ink ({target_sym:.4f}), require MR_score≥0 (corr≤0), same Wz range …"
    )
    ranked_b = grid_match_sym(
        df_osm, target_sym, args.thresh, wz_min=8, wz_max=args.grid_max_wz, ws_min=2, min_mr=0.0
    )
    if not ranked_b:
        print("  (no windows with MR_score≥0 and valid buckets)")
    else:
        for i, (err, wz, ws) in enumerate(ranked_b[: args.grid_top], 1):
            m = metrics_for_window(df_osm, wz, ws, args.thresh)
            assert m is not None
            print(
                f"  {i:2d}. Wz={wz:3d} Ws={ws:3d}  sym_f1={m['sym_f1']:.4f}  MR_score={m['mr_score']:+.4f}  err²={err:.2e}"
            )
    best = ranked[0]
    m_best = metrics_for_window(df_osm, best[1], best[2], args.thresh)
    assert m_best is not None
    print()
    print("--- Takeaway ---")
    print(
        "MR_score = **-corr(signal, f1)**.  Positive ⇒ high smoothed z tends to coincide with "
        "a **down** tick next (fade-the-stretch).  **sym_f1** sums |conditional mean next-tick moves| at ±threshold."
    )
    print(
        "At jmerle mid, **short windows** on ink can show near-zero or wrong-signed corr; osmium is usually stronger there. "
        f"At **({args.ref_wz},{args.ref_ws})**, ink sym_f1 is **larger** than osmium for the same window — "
        "to match ink's **sym_f1**, see Grid B; to match ink's **MR_score** only, see Grid A."
    )


if __name__ == "__main__":
    main()
