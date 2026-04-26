#!/usr/bin/env python3
"""How many v26 triggers have post-window activity crossing merged day3 start (2e6)?"""
from __future__ import annotations

import json
from pathlib import Path

SIG = Path(__file__).resolve().parent / "outputs" / "r4_v26_signals.json"
DAY3 = 2_000_000


def main() -> None:
    obj = json.loads(SIG.read_text())
    trigs = sorted(int(x) for x in obj["mark67_extract_buy_aggr_filtered_merged_ts"])
    W = int(obj["window_ts"])
    cross = []
    for T in trigs:
        if T < DAY3 <= T + W:
            cross.append(T)
    lines = [
        f"W={W} day3_start={DAY3}",
        f"n_triggers_total={len(trigs)}",
        f"n_triggers_with_T_lt_day3_and_T_plus_W_ge_day3={len(cross)}",
        "merged_T values:",
        *[f"  {t}" for t in cross],
    ]
    outp = Path(__file__).resolve().parent / "outputs" / "r4_p10_windows_crossing_into_day3.txt"
    outp.write_text("\n".join(lines) + "\n")
    print("wrote", outp, "n_cross", len(cross))


if __name__ == "__main__":
    main()
