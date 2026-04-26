#!/usr/bin/env python3
"""
For precomputed signal lists, compute **joint Sonic tight** at **fire** time
(print_ts + LAG on the same tape day), matching trader lag convention.

Outputs: outputs/phase3/act_gate_tight_at_fire_coverage.json and per-set CSV rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def joint_tight_at(px: pd.DataFrame, tape_day: int, ts: int) -> bool | None:
    a = _spread_at(px, tape_day, ts, VEV_5200)
    b = _spread_at(px, tape_day, ts, VEV_5300)
    if a is None or b is None:
        return None
    return a <= TH and b <= TH


def load_all_prices() -> pd.DataFrame:
    frames = []
    for d in (1, 2, 3):
        p = DATA / f"prices_round_4_day_{d}.csv"
        if p.is_file():
            frames.append(pd.read_csv(p, sep=";"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    px = load_all_prices()
    sets = [
        (OUT_DIR / "signals_mark67_to_mark22_extract_joint_tight_at_print.json", "mark67_to_mark22_tight_print"),
        (OUT_DIR / "signals_mark67_to_mark49_extract_joint_tight_at_print.json", "mark67_to_mark49_tight_print"),
    ]
    summ: list[dict] = []
    for path, lab in sets:
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for item in raw:
            tape_day, local_ts = int(item[0]), int(item[1])
            fire_ts = local_ts + LAG
            t = joint_tight_at(px, tape_day, fire_ts)
            rows.append(
                {
                    "signal_set": lab,
                    "tape_day": tape_day,
                    "print_ts": local_ts,
                    "fire_ts": fire_ts,
                    "joint_tight_at_fire": t,
                }
            )
        df = pd.DataFrame(rows)
        csv_path = OUT_DIR / f"act_gate_at_fire_{lab}.csv"
        df.to_csv(csv_path, index=False)
        ok = df["joint_tight_at_fire"] == True
        bad = df["joint_tight_at_fire"].isna()
        summ.append(
            {
                "label": lab,
                "n": int(len(df)),
                "n_tight_at_fire": int(ok.sum()),
                "p_tight_at_fire": float(ok.mean()) if len(df) else 0.0,
                "n_missing_book": int(bad.sum()),
                "detail_csv": str(csv_path),
            }
        )

    out_path = OUT_DIR / "act_gate_tight_at_fire_coverage.json"
    out_path.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
