#!/usr/bin/env python3
"""
Build per-timestamp boolean: Sonic joint tight AND within ±5 ticks (500 time units) of a
Mark 01 → Mark 22 print on VEV_5300 (tape truth).

Output: manual_traders/R4/r3v_jump_gap_filter_17/precomputed/r4_m0122_5300_window_joint_signal.json
Shape: {"1": [ts, ...], "2": [...], "3": [...]} — sorted unique timestamps where signal is True.

Run once from repo root:
  python3 manual_traders/R4/r3v_jump_gap_filter_17/preprocess_r4_m0122_5300_gate_signal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT_DIR = Path(__file__).resolve().parent / "precomputed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "r4_m0122_5300_window_joint_signal.json"

DAYS = (1, 2, 3)
WINDOW_TICKS = 5
TICK = 100
TH = 2
MAX_TS = 999900


def strip(df: pd.DataFrame, product: str) -> pd.DataFrame:
    v = (
        df[df["product"] == product]
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
    )
    bid = pd.to_numeric(v["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(v["ask_price_1"], errors="coerce")
    sp = (ask - bid).astype(int)
    return pd.DataFrame({"timestamp": v["timestamp"].astype(int), "spread": sp})


def joint_series(pr: pd.DataFrame) -> pd.Series:
    a = strip(pr, "VEV_5200").rename(columns={"spread": "s52"})
    b = strip(pr, "VEV_5300").rename(columns={"spread": "s53"})
    m = a.merge(b, on="timestamp", how="inner")
    m["tight"] = (m["s52"] <= TH) & (m["s53"] <= TH)
    return m.set_index("timestamp")["tight"]


def main() -> None:
    out: dict[str, list[int]] = {}
    for day in DAYS:
        pr = pd.read_csv(DATA / f"prices_round_4_day_{day}.csv", sep=";")
        tr = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        prints = tr[
            (tr["buyer"] == "Mark 01")
            & (tr["seller"] == "Mark 22")
            & (tr["symbol"] == "VEV_5300")
        ]["timestamp"].astype(int).unique()
        tight = joint_series(pr)
        ts_all = sorted(pr["timestamp"].unique())
        sig: set[int] = set()
        for ts in ts_all:
            if not bool(tight.get(ts, False)):
                continue
            ok = False
            for p in prints:
                if abs(int(ts) - int(p)) <= WINDOW_TICKS * TICK:
                    ok = True
                    break
            if ok:
                sig.add(int(ts))
        out[str(day)] = sorted(sig)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print("Wrote", OUT, "counts:", {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
