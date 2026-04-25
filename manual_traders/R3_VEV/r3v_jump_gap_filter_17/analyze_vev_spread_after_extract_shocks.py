"""
Round-3 tape: top-of-book spread (ask_1 - bid_1) on each VEV vs extract |dS|.

Jump/shock row: |dS_extract| >= 3 (same threshold as traders).
Compare spread distribution shock vs non-shock, and mean next-tick spread change.

Output JSON for v18: which strikes widen relatively more (for OTM-aware widening).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_3"
EX = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV = [f"VEV_{k}" for k in STRIKES]
JUMP = 3.0
COLS = ["timestamp", "product", "bid_price_1", "ask_price_1"]


def main() -> None:
    by_strike: dict[str, dict] = {}
    pooled_shock_sp = []
    pooled_calm_sp = []

    for day in (0, 1, 2):
        path = DATA / f"prices_round_3_day_{day}.csv"
        df = pd.read_csv(path, sep=";", usecols=COLS)
        df = df[df["product"].isin([EX] + VEV)]
        pvt = df.pivot_table(
            index="timestamp", columns="product", values=["bid_price_1", "ask_price_1"], aggfunc="last"
        )
        if EX not in pvt.columns.get_level_values(1):
            continue
        s = pvt["bid_price_1"][EX].astype(float)
        # mid from bid/ask if we had mid column — use average of bid ask for extract
        aex = pvt["ask_price_1"][EX].astype(float)
        smid = 0.5 * (s + aex)
        dS = smid.diff().abs().fillna(0.0)
        shock = (dS >= JUMP).astype(int)

        for sym in VEV:
            if sym not in pvt["bid_price_1"].columns:
                continue
            bid = pvt["bid_price_1"][sym].astype(float)
            ask = pvt["ask_price_1"][sym].astype(float)
            sp = (ask - bid).clip(lower=0)
            nxt = sp.shift(-1)
            dnext = (nxt - sp).where(shock == 1)

            calm = sp[shock == 0]
            sh = sp[shock == 1]
            if len(calm) > 10 and len(sh) > 0:
                r_med = float(sh.median()) / max(1e-6, float(calm.median()))
                r_p90 = float(sh.quantile(0.9)) / max(1e-6, float(calm.quantile(0.9)))
                r_p95 = float(sh.quantile(0.95)) / max(1e-6, float(calm.quantile(0.95)))
            else:
                r_med = float("nan")
                r_p90 = float("nan")
                r_p95 = float("nan")
            dn = dnext.dropna()
            mean_dnext = float(dn.mean()) if len(dn) else float("nan")

            k = sym.split("_")[1]
            if k not in by_strike:
                by_strike[k] = {
                    "ratio_median_shock_over_calm": [],
                    "ratio_p90_shock_over_calm": [],
                    "ratio_p95_shock_over_calm": [],
                    "mean_dspread_next_on_shock": [],
                }
            by_strike[k]["ratio_median_shock_over_calm"].append(r_med)
            by_strike[k]["ratio_p90_shock_over_calm"].append(r_p90)
            by_strike[k]["ratio_p95_shock_over_calm"].append(r_p95)
            by_strike[k]["mean_dspread_next_on_shock"].append(mean_dnext)

            pooled_shock_sp.extend(sh.dropna().tolist())
            pooled_calm_sp.extend(calm.dropna().tolist())

    out_strikes: dict[str, dict] = {}
    for k, v in by_strike.items():
        rm = [x for x in v["ratio_median_shock_over_calm"] if np.isfinite(x)]
        r9 = [x for x in v["ratio_p90_shock_over_calm"] if np.isfinite(x)]
        r95 = [x for x in v["ratio_p95_shock_over_calm"] if np.isfinite(x)]
        md = [x for x in v["mean_dspread_next_on_shock"] if np.isfinite(x)]
        out_strikes[k] = {
            "median_ratio_shock_over_calm_days_pooled": float(np.nanmedian(rm)) if rm else None,
            "median_p90_ratio_shock_over_calm_across_days": float(np.nanmedian(r9)) if r9 else None,
            "median_p95_ratio_shock_over_calm_across_days": float(np.nanmedian(r95)) if r95 else None,
            "mean_next_tick_spread_delta_on_shock": float(np.nanmean(md)) if md else None,
        }

    summ = {
        "jump_threshold_abs_dS": JUMP,
        "pooled_median_spread_shock": float(np.median(pooled_shock_sp)) if pooled_shock_sp else None,
        "pooled_median_spread_calm": float(np.median(pooled_calm_sp)) if pooled_calm_sp else None,
        "per_strike": out_strikes,
    }
    outp = Path(__file__).resolve().parent / "analysis_vev_spread_after_shocks.json"
    outp.write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")
    print(outp)
    print(json.dumps(summ, indent=2)[:4000])


if __name__ == "__main__":
    main()
