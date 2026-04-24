"""
VEV_5200-only: distribution of switch_mean and raw theo_diff under Frankfurt formulas
(global smile coeffs). Writes threshold_suggestions.txt (quantiles) — does not auto-edit calibration.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent.parent
_COMBINED = REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"
sys.path.insert(0, str(_COMBINED))

from plot_iv_smile_round3 import t_years_effective  # noqa: E402

from frankfurt_iv_scalp_core import (  # noqa: E402
    book_from_row,
    compute_option_indicators,
    load_calibration,
    synthetic_walls_if_missing,
)

DATA = REPO / "Prosperity4Data" / "ROUND_3"
CAL = Path(__file__).resolve().parent / "calibration.json"
OUT = Path(__file__).resolve().parent / "threshold_suggestions.txt"
STEP = 50
K = 5200
OPT = "VEV_5200"
U = "VELVETFRUIT_EXTRACT"


def main() -> None:
    cal = load_calibration(CAL)
    ema: dict[str, float] = {}
    sw_list: list[float] = []
    diff_list: list[float] = []

    for day in (0, 1, 2):
        ema.clear()
        path = DATA / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(path, sep=";")
        ts_list = sorted(df["timestamp"].unique())[::STEP]
        for ts in ts_list:
            g = df[df["timestamp"] == ts]
            if OPT not in g["product"].values or U not in g["product"].values:
                continue
            ro = g[g["product"] == OPT].iloc[0].to_dict()
            ru = g[g["product"] == U].iloc[0].to_dict()
            _, _, bid_wall, ask_wall, best_bid, best_ask, wall_mid = book_from_row(ro)
            _, _, _, _, ubb, uba, _uwm = book_from_row(ru)
            if ubb is None or uba is None:
                continue
            u_mid = 0.5 * float(ubb) + 0.5 * float(uba)
            bid_wall, ask_wall, wall_mid, best_bid, best_ask = synthetic_walls_if_missing(
                bid_wall, ask_wall, best_bid, best_ask
            )
            if wall_mid is None or best_bid is None or best_ask is None:
                continue
            T = t_years_effective(day, int(ts))
            ind = compute_option_indicators(cal, ema, u_mid, K, T, wall_mid, best_bid, best_ask, OPT)
            if ind.get("switch_mean") is not None:
                sw_list.append(float(ind["switch_mean"]))
            if ind.get("current_theo_diff") is not None:
                diff_list.append(float(ind["current_theo_diff"]))

    sw = np.asarray(sw_list, float)
    dd = np.asarray(diff_list, float)
    lines = [
        "VEV_5200 threshold suggestions (Frankfurt-style stats on historical ROUND_3)",
        f"n_switch={len(sw)} n_diff={len(dd)}",
        "",
        "switch_mean quantiles (IV_SCALPING_THR is compared to this):",
    ]
    for q in (0.5, 0.7, 0.8, 0.9, 0.95):
        lines.append(f"  p{int(q*100)}: {float(np.quantile(sw, q)):.6f}")
    lines.extend(["", "theo_diff (wall_mid - theo) abs quantiles:", ""])
    for q in (0.5, 0.9, 0.95):
        lines.append(f"  |diff| p{int(q*100)}: {float(np.quantile(np.abs(dd), q)):.6f}")
    lines.extend(
        [
            "",
            "Frankfurt defaults in calibration.json: IV_SCALPING_THR=0.7, THR_OPEN=0.5, THR_CLOSE=0.",
            "Tune if your p70 on switch_mean is far from 0.7.",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
