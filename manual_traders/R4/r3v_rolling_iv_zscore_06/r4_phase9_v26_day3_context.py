#!/usr/bin/env python3
"""Merged **day 3** subset of v26 triggers: spacing and tape forward V5300 at print."""
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
    obj = json.loads(SIG.read_text())
    triggers = sorted(int(x) for x in obj["mark67_extract_buy_aggr_filtered_merged_ts"])
    cum = {int(k): int(v) for k, v in obj["day_cum_offset"].items()}
    d3_off = cum[3]

    day3 = [t for t in triggers if t >= d3_off]
    lines = [
        f"v26 triggers on merged day 3 (T>={d3_off}): n={len(day3)}",
    ]
    if len(day3) >= 2:
        gaps = [day3[i + 1] - day3[i] for i in range(len(day3) - 1)]
        lines.append(f"gap_merged_start_to_start: min={min(gaps)} max={max(gaps)} mean={sum(gaps)/len(gaps):.0f}")

    px = load_px()
    rows = []
    for T in day3:
        t_local = T - d3_off
        ts53, m53 = series_pair(px, 3, "VEV_5300")
        rows.append(
            {
                "merged_T": T,
                "tape_ts": t_local,
                "fwd5300_K5": forward_delta(ts53, m53, int(t_local), 5),
                "fwd5300_K20": forward_delta(ts53, m53, int(t_local), 20),
            }
        )
    ev = pd.DataFrame(rows)
    if not ev.empty:
        a = ev["fwd5300_K5"].to_numpy()
        b = ev["fwd5300_K20"].to_numpy()
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        lines.append(
            f"fwd5300_at_print: meanK5={a.mean():.4f} fracposK5={(a>0).mean():.3f} meanK20={b.mean():.4f} fracposK20={(b>0).mean():.3f}"
        )
        lines.append("per_trigger merged_T tape_ts fwdK5 fwdK20:")
        for _, r in ev.iterrows():
            lines.append(
                f"  {int(r['merged_T'])}\t{int(r['tape_ts'])}\t{r['fwd5300_K5']:.4g}\t{r['fwd5300_K20']:.4g}"
            )

    outp = OUT / "r4_p9_v26_triggers_day3_detail.txt"
    outp.write_text("\n".join(lines) + "\n")
    print("wrote", outp)


if __name__ == "__main__":
    main()
