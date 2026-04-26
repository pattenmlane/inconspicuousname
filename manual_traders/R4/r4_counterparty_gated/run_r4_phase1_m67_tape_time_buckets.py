"""
Supplement: Mark 67 aggressive-buy on VELVETFRUIT_EXTRACT — mean fwd mid K=5 by tape-time
decile (decile = min(9, timestamp // 100000); R4 tape step 100, max ts 999900).

Rerun: python3 run_r4_phase1_m67_tape_time_buckets.py
Outputs:
  - analysis_outputs/r4_m67_extract_fwd5_by_tape_decile.csv
  - analysis_outputs/r4_m67_extract_fwd5_by_tape_decile_pooled.csv
  - analysis_outputs/r4_m67_tape_time_bucket_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
K = 5
TARGET = "VELVETFRUIT_EXTRACT"
M67 = "Mark 67"


def load_prices() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA.glob("prices_round_4_day_*.csv")):
        day = int(p.stem.replace("prices_round_4_day_", ""))
        df = pd.read_csv(p, sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_trades() -> pd.DataFrame:
    frames = []
    for p in sorted(DATA.glob("trades_round_4_day_*.csv")):
        day = int(p.stem.replace("trades_round_4_day_", ""))
        df = pd.read_csv(p, sep=";")
        df["day"] = day
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def idx_at_or_before(ts_arr: np.ndarray, t: int) -> int:
    i = int(np.searchsorted(ts_arr, t, side="right") - 1)
    return max(0, min(i, len(ts_arr) - 1))


def classify_aggression(price: float, bid1: float, ask1: float) -> str:
    if not (np.isfinite(price) and np.isfinite(bid1) and np.isfinite(ask1)):
        return "unknown"
    if ask1 - bid1 <= 0:
        return "unknown"
    if price >= ask1 - 1e-9:
        return "aggr_buy"
    if price <= bid1 + 1e-9:
        return "aggr_sell"
    return "passive_mid"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = load_prices()
    tr = load_trades()
    tr["price"] = pd.to_numeric(tr["price"], errors="coerce")
    px2 = px.copy()
    px2["product"] = px2["product"].astype(str)
    fwd: dict = {}
    bidask: dict = {}
    for (day, sym), g in px2.groupby(["day", "product"]):
        g = g.sort_values("timestamp")
        ts = g["timestamp"].to_numpy(dtype=np.int64)
        mid = pd.to_numeric(g["mid_price"], errors="coerce").to_numpy(dtype=float)
        bp = pd.to_numeric(g["bid_price_1"], errors="coerce").to_numpy(dtype=float)
        ap = pd.to_numeric(g["ask_price_1"], errors="coerce").to_numpy(dtype=float)
        k = (int(day), str(sym))
        fwd[k] = {"ts": ts, "mid": mid}
        bidask[k] = {"ts": ts, "bid1": bp, "ask1": ap}

    rows: list[dict] = []
    for _, r in tr.iterrows():
        day, sym, ts = int(r["day"]), str(r["symbol"]), int(r["timestamp"])
        if sym != TARGET or str(r["buyer"]) != M67:
            continue
        key = (day, sym)
        if key not in fwd or key not in bidask:
            continue
        b = bidask[key]
        i0 = idx_at_or_before(b["ts"], ts)
        if classify_aggression(float(r["price"]), float(b["bid1"][i0]), float(b["ask1"][i0])) != "aggr_buy":
            continue
        s = fwd[key]
        ts_arr, mid = s["ts"], s["mid"]
        i = idx_at_or_before(ts_arr, ts)
        j = i + K
        if j >= len(mid):
            continue
        dec = int(np.clip(ts // 100_000, 0, 9))
        rows.append(
            {
                "day": day,
                "timestamp": ts,
                "tape_decile": dec,
                f"fwd_mid_{K}": float(mid[j]) - float(mid[i]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        (OUT / "r4_m67_extract_fwd5_by_tape_decile.csv").write_text("day,tape_decile,n,mean_fwd_mid_5\n", encoding="utf-8")
        return
    g = (
        df.groupby(["day", "tape_decile"])[f"fwd_mid_{K}"]
        .agg(n="count", mean_fwd_mid_5="mean")
        .reset_index()
    )
    g.to_csv(OUT / "r4_m67_extract_fwd5_by_tape_decile.csv", index=False)
    pool = (
        df.groupby("tape_decile")[f"fwd_mid_{K}"]
        .agg(n="count", mean_fwd_mid_5="mean")
        .reset_index()
    )
    pool.to_csv(OUT / "r4_m67_extract_fwd5_by_tape_decile_pooled.csv", index=False)
    (OUT / "r4_m67_tape_time_bucket_summary.json").write_text(
        json.dumps(
            {
                "horizon_K": K,
                "horizon_unit": "K steps along price row index for (day, VELVETFRUIT_EXTRACT) — same as run_r4_phase1_analysis.py",
                "timestamp_unit": "tape step index; step 100 from 0 to 999900 per day in ROUND_4 sample",
                "tape_decile": "int clip(timestamp // 100000, 0, 9) — 10 ~equal-width buckets in tape time",
                "n_m67_extract_aggr_buy": int(len(df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote", OUT, "M67 by tape decile, n =", len(df))


if __name__ == "__main__":
    main()
