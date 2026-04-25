#!/usr/bin/env python3
"""
ROUND_3 tapes: at each row timestamp, pick nearest-ATM VEV to extract mid; record
quoted width = best_ask - best_bid (from CSV bid_price_1 / ask_price_1).

Optional Greek: ATM vega from Frankfurt smile + get_option_values (same as traders).

Output: analysis_atm_vev_spread.json with per-day and pooled quantiles.
TTE: plot_iv_smile_round3.t_years_effective(csv_day, timestamp).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "round3work" / "voucher_work" / "5200_work"))
sys.path.insert(0, str(REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"))

from frankfurt_iv_scalp_core import get_option_values, load_calibration  # noqa: E402
from plot_iv_smile_round3 import t_years_effective  # noqa: E402

DATA = REPO / "Prosperity4Data" / "ROUND_3"
CAL_PATH = REPO / "round3work" / "voucher_work" / "5200_work" / "calibration.json"
OUT = Path(__file__).resolve().parent / "analysis_atm_vev_spread.json"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]


def main() -> None:
    cal = load_calibration(CAL_PATH)
    coeffs = cal["coeffs_high_to_low"]
    by_day: dict[int, dict] = {}
    all_widths: list[float] = []
    all_vega: list[float] = []

    for csv_day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{csv_day}.csv"
        df = pd.read_csv(path, sep=";")
        ex = df[df["product"] == "VELVETFRUIT_EXTRACT"].set_index("timestamp")["mid_price"]

        rows_out: list[dict] = []
        for ts, group in df.groupby("timestamp"):
            if ts not in ex.index:
                continue
            S = float(ex.loc[ts])
            k_star = min(STRIKES, key=lambda k: abs(float(k) - S))
            sym = f"VEV_{k_star}"
            sub = group[group["product"] == sym]
            if sub.empty:
                continue
            r = sub.iloc[0]
            bp = r.get("bid_price_1")
            ap = r.get("ask_price_1")
            if pd.isna(bp) or pd.isna(ap):
                continue
            bb, ba = int(bp), int(ap)
            if ba <= bb:
                continue
            w = float(ba - bb)
            T = float(t_years_effective(csv_day, int(ts)))
            theo, _d, vega = get_option_values(S, float(k_star), T, coeffs)
            vg = float(vega) if theo == theo and vega == vega else float("nan")
            rows_out.append({"width": w, "vega": vg})

        w_arr = np.array([x["width"] for x in rows_out], dtype=float)
        v_arr = np.array([x["vega"] for x in rows_out], dtype=float)
        by_day[csv_day] = {
            "n": int(len(w_arr)),
            "width_p50": float(np.nanpercentile(w_arr, 50)),
            "width_p75": float(np.nanpercentile(w_arr, 75)),
            "width_p90": float(np.nanpercentile(w_arr, 90)),
            "width_p95": float(np.nanpercentile(w_arr, 95)),
            "vega_p50": float(np.nanpercentile(v_arr[np.isfinite(v_arr)], 50)) if np.any(np.isfinite(v_arr)) else float("nan"),
        }
        all_widths.extend(w_arr.tolist())
        all_vega.extend(v_arr[np.isfinite(v_arr)].tolist())

    aw = np.array(all_widths, dtype=float)
    av = np.array(all_vega, dtype=float)
    summary = {
        "by_csv_day": by_day,
        "pooled": {
            "n": int(len(aw)),
            "width_p50": float(np.nanpercentile(aw, 50)),
            "width_p75": float(np.nanpercentile(aw, 75)),
            "width_p90": float(np.nanpercentile(aw, 90)),
            "width_p95": float(np.nanpercentile(aw, 95)),
            "vega_p50": float(np.nanpercentile(av, 50)) if len(av) else float("nan"),
        },
        "method": "width=ask1-bid1 on nearest ATM VEV; vega=get_option_values ATM; T=t_years_effective",
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
