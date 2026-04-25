#!/usr/bin/env python3
"""
Round 3 tapes: relate short-horizon changes in VELVETFRUIT_EXTRACT to ATM-ish implied vol (VEV_5200).
r = corr(d log S, d IV) on subsampled consecutive rows per day; T from t_years_effective(day, ts).
Methodology for analysis.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "iv5200_vs_extract_logdiff.json"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _r3v_smile_core import implied_vol_bisect, t_years_effective  # noqa: E402

U = "VELVETFRUIT_EXTRACT"
OPT = "VEV_5200"
K0 = 5200


def main() -> None:
    out: dict = {}
    for day in (0, 1, 2):
        df = pd.read_csv(DATA / f"prices_round_3_day_{day}.csv", sep=";")
        pvt = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
        if U not in pvt.columns or OPT not in pvt.columns:
            continue
        ts_idx = pvt.index.tolist()
        dlog_s: list[float] = []
        div: list[float] = []
        for t_prev, t in zip(ts_idx, ts_idx[1:]):
            S0, S1 = pvt.at[t_prev, U], pvt.at[t, U]
            m0, m1 = pvt.at[t_prev, OPT], pvt.at[t, OPT]
            if pd.isna(S0) or pd.isna(S1) or S0 <= 0 or S1 <= 0:
                continue
            if pd.isna(m0) or pd.isna(m1):
                continue
            T0 = t_years_effective(day, int(t_prev))
            T1 = t_years_effective(day, int(t))
            iv0 = implied_vol_bisect(float(m0), float(S0), float(K0), T0)
            iv1 = implied_vol_bisect(float(m1), float(S1), float(K0), T1)
            if iv0 is None or iv1 is None:
                continue
            dlog_s.append(math.log(S1) - math.log(S0))
            div.append(float(iv1) - float(iv0))
        if len(dlog_s) < 20:
            out[str(day)] = {"n": len(dlog_s), "corr": None}
        else:
            c = float(np.corrcoef(dlog_s, div)[0, 1])
            out[str(day)] = {"n": len(dlog_s), "corr_logS_div": c}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": f"Consecutive {OPT} row pairs per day: d(log {U}) vs d(IV) where IV = bisection from mid, T=t_years_effective(day,ts).",
        "by_csv_day": out,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
