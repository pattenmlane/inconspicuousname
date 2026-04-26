#!/usr/bin/env python3
"""
Filter phase2 Mark67 buy-aggr extract signals to rows where **joint Sonic tight**
holds on the **price tape** at fire time = print_ts + LAG (same LAG=100 as traders).

Reads:  outputs/phase2/signals_mark67_buy_aggr_extract.json
Writes: outputs/phase3/signals_mark67_buy_aggr_extract_tight_at_fire.json

Run: python3 manual_traders/R4/r3v_wing_vs_core_spread_04/r4_filter_mark67_signals_tight_at_fire.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
IN_PATH = Path(__file__).resolve().parent / "outputs" / "phase2" / "signals_mark67_buy_aggr_extract.json"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "signals_mark67_buy_aggr_extract_tight_at_fire.json"

LAG = 100
TH = 2
VEV_5200, VEV_5300 = "VEV_5200", "VEV_5300"


def _spread_at(px: pd.DataFrame, tape_day: int, ts: int, prod: str) -> float | None:
    r = px[(px["day"] == tape_day) & (px["timestamp"] == ts) & (px["product"] == prod)]
    if len(r) != 1:
        return None
    row = r.iloc[0]
    try:
        b = float(row["bid_price_1"])
        a = float(row["ask_price_1"])
    except (TypeError, ValueError, KeyError):
        return None
    return float(a - b)


def joint_tight_at(px: pd.DataFrame, tape_day: int, ts: int) -> bool:
    a = _spread_at(px, tape_day, ts, VEV_5200)
    b = _spread_at(px, tape_day, ts, VEV_5300)
    if a is None or b is None:
        return False
    return a <= TH and b <= TH


def main() -> None:
    raw = json.loads(IN_PATH.read_text(encoding="utf-8"))
    frames = []
    for d in (1, 2, 3):
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            frames.append(pd.read_csv(p, sep=";"))
    px = pd.concat(frames, ignore_index=True)

    kept: list[list[int]] = []
    for tape_day, local_ts in raw:
        d, ts = int(tape_day), int(local_ts)
        fire_ts = ts + LAG
        if joint_tight_at(px, d, fire_ts):
            kept.append([d, ts])

    OUT_PATH.write_text(json.dumps(kept), encoding="utf-8")
    meta = {
        "source": str(IN_PATH),
        "n_in": len(raw),
        "n_out": len(kept),
        "p_kept": len(kept) / len(raw) if raw else 0.0,
        "LAG": LAG,
        "TH": TH,
        "output": str(OUT_PATH),
    }
    (OUT_DIR / "mark67_buy_aggr_filter_tight_at_fire_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
