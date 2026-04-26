#!/usr/bin/env python3
"""At each v26 trigger timestamp: extract BBO spread vs forward VEV_5300 mid K=5/K=20."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs"
SIG = OUT / "r4_v26_signals.json"
DAYS = [1, 2, 3]


def load_px() -> pd.DataFrame:
    fs = []
    for d in DAYS:
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            df = pd.read_csv(p, sep=";")
            df["day"] = d
            fs.append(df)
    return pd.concat(fs, ignore_index=True)


def series_pair(px: pd.DataFrame, day: int, sym: str) -> tuple[np.ndarray, np.ndarray]:
    g = px[(px["day"] == day) & (px["product"] == sym)].sort_values("timestamp")
    return g["timestamp"].to_numpy(dtype=np.int64), g["mid_price"].astype(float).to_numpy()


def forward_delta(ts: np.ndarray, mid: np.ndarray, t0: int, k: int) -> float:
    i = int(np.searchsorted(ts, t0, side="left"))
    if i >= len(ts):
        return float("nan")
    if ts[i] != t0:
        i = int(np.searchsorted(ts, t0, side="right") - 1)
    i = max(0, min(i, len(ts) - 1))
    j = min(i + k, len(mid) - 1)
    if j <= i:
        return float("nan")
    return float(mid[j] - mid[i])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obj = json.loads(SIG.read_text())
    triggers = sorted(int(x) for x in obj["mark67_extract_buy_aggr_filtered_merged_ts"])
    cum = {int(k): int(v) for k, v in obj["day_cum_offset"].items()}

    def merged_to_day_ts(t: int) -> tuple[int, int]:
        for d in sorted(cum.keys(), reverse=True):
            if t >= cum[d]:
                return d, int(t - cum[d])
        return 1, int(t)

    px = load_px()
    ex = px[px["product"] == "VELVETFRUIT_EXTRACT"].copy()
    ex["sp"] = ex["ask_price_1"].astype(float) - ex["bid_price_1"].astype(float)
    ex_map = ex.set_index(["day", "timestamp"])["sp"].to_dict()

    rows = []
    for T in triggers:
        d, t0 = merged_to_day_ts(T)
        spx = float(ex_map.get((d, t0), float("nan")))
        ts53, m53 = series_pair(px, d, "VEV_5300")
        rows.append(
            {
                "day": d,
                "extract_spread": spx,
                "fwd5300_K5": forward_delta(ts53, m53, t0, 5),
                "fwd5300_K20": forward_delta(ts53, m53, t0, 20),
            }
        )
    ev = pd.DataFrame(rows)
    ev = ev.dropna(subset=["extract_spread"])

    def summ(mask: pd.Series, name: str) -> str:
        g = ev[mask]
        if g.empty:
            return f"{name}: n=0"
        a = g["fwd5300_K5"].to_numpy()
        b = g["fwd5300_K20"].to_numpy()
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        return (
            f"{name}: n={len(g)} ext_spread_mean={g['extract_spread'].mean():.3f} "
            f"meanK5={a.mean():.4f} fracposK5={(a>0).mean():.3f} meanK20={b.mean():.4f} fracposK20={(b>0).mean():.3f}"
        )

    lines = [
        "v26 triggers (n=%d): extract spread at print vs forward V5300 mid." % len(ev),
        summ(ev["extract_spread"] <= 2, "extract_spread<=2"),
        summ(ev["extract_spread"] > 2, "extract_spread>2"),
        "--- by day (extract<=2) ---",
    ]
    for d in sorted(ev["day"].unique()):
        lines.append(summ((ev["day"] == d) & (ev["extract_spread"] <= 2), f"day_{d}_ext<=2"))

    outp = OUT / "r4_p7_extract_spread_at_v26_triggers.txt"
    outp.write_text("\n".join(lines) + "\n")
    print("wrote", outp)


if __name__ == "__main__":
    main()
