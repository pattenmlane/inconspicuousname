"""
Phase-1 supplement: passive liquidity-provider markouts (counterparty on non-aggressive side).

When aggression == aggr_buy, the seller sat on the ask (passive seller) — tag seller U.
When aggression == aggr_sell, the buyer sat on the bid (passive buyer) — tag buyer U.

For each (U_passive_role, symbol, K) with n>=20: mean, median, t-stat, bootstrap 95% CI for mean,
frac positive on fwd_mid_K (same symbol tape step).

Also: Mark 67 aggressive extract buy stratified by (day, ts_bucket, spread_bucket, burst_ge4), n>=10.

Outputs:
  r4_passive_seller_after_aggr_buy_markout_by_name.csv
  r4_passive_buyer_after_aggr_sell_markout_by_name.csv
  r4_stratified_mark67_extract_by_day_session_spread_burst.csv
  r4_phase1_passive_lp_summary.json

Rerun: python3 run_r4_phase1_passive_lp_markouts.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EV = Path(__file__).resolve().parent / "analysis_outputs" / "r4_phase1_trade_events.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
HORIZONS = (5, 20, 100)
N_MIN = 20
N_MIN_M67 = 10


def safe_tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    if s < 1e-12:
        return float("nan")
    return m / (s / math.sqrt(n))


def boot_ci(x: np.ndarray, seed: int = 0) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(2000, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def agg_cells(df: pd.DataFrame, name_col: str, role_label: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (u, sym), g in df.groupby([name_col, "symbol"]):
        if not u or str(u) == "nan":
            continue
        for K in HORIZONS:
            col = f"fwd_mid_{K}"
            x = g[col].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < N_MIN:
                continue
            lo, hi = boot_ci(x, seed=hash((u, sym, K, role_label)) % (2**31 - 1))
            rows.append(
                {
                    "passive_name": str(u),
                    "role": role_label,
                    "symbol": str(sym),
                    "K": K,
                    "n": len(x),
                    "mean_fwd": float(np.mean(x)),
                    "median_fwd": float(np.median(x)),
                    "t_stat": safe_tstat(x),
                    "frac_pos": float(np.mean(x > 0)),
                    "boot_ci95_mean_lo": lo,
                    "boot_ci95_mean_hi": hi,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("t_stat", key=lambda s: s.abs(), ascending=False)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(EV)

    buy = ev[ev["aggression"] == "aggr_buy"].copy()
    buy["passive_seller"] = buy["seller"].astype(str)
    ps = agg_cells(buy, "passive_seller", "passive_seller_after_aggr_buy")
    ps.to_csv(OUT / "r4_passive_seller_after_aggr_buy_markout_by_name.csv", index=False)

    sell = ev[ev["aggression"] == "aggr_sell"].copy()
    sell["passive_buyer"] = sell["buyer"].astype(str)
    pb = agg_cells(sell, "passive_buyer", "passive_buyer_after_aggr_sell")
    pb.to_csv(OUT / "r4_passive_buyer_after_aggr_sell_markout_by_name.csv", index=False)

    m67 = ev[
        (ev["buyer"] == "Mark 67")
        & (ev["aggression"] == "aggr_buy")
        & (ev["symbol"] == "VELVETFRUIT_EXTRACT")
    ].copy()
    mrows: list[dict] = []
    for (d, tsb, spr, br), g in m67.groupby(["day", "ts_bucket", "spread_bucket", "burst_ge4"]):
        for K in HORIZONS:
            col = f"fwd_mid_{K}"
            x = g[col].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < N_MIN_M67:
                continue
            lo, hi = boot_ci(x, seed=int(d) * 1000 + int(br) * 10 + K)
            mrows.append(
                {
                    "day": int(d),
                    "ts_bucket": int(tsb) if pd.notna(tsb) else -1,
                    "spread_bucket": str(spr),
                    "burst_ge4": int(br),
                    "K": K,
                    "n": len(x),
                    "mean_fwd": float(np.mean(x)),
                    "median_fwd": float(np.median(x)),
                    "t_stat": safe_tstat(x),
                    "frac_pos": float(np.mean(x > 0)),
                    "boot_ci95_mean_lo": lo,
                    "boot_ci95_mean_hi": hi,
                }
            )
    pd.DataFrame(mrows).to_csv(OUT / "r4_stratified_mark67_extract_by_day_session_spread_burst.csv", index=False)

    worst_ps = ps.nsmallest(8, "mean_fwd") if not ps.empty else pd.DataFrame()
    worst_pb = pb.nsmallest(8, "mean_fwd") if not pb.empty else pd.DataFrame()
    summ = {
        "n_passive_seller_cells_n20": int(len(ps)),
        "n_passive_buyer_cells_n20": int(len(pb)),
        "worst_passive_seller_K20_head": worst_ps[worst_ps["K"] == 20].head(5).to_dict(orient="records")
        if not worst_ps.empty
        else [],
        "worst_passive_buyer_K20_head": worst_pb[worst_pb["K"] == 20].head(5).to_dict(orient="records")
        if not worst_pb.empty
        else [],
        "n_m67_stratified_rows": len(mrows),
    }
    (OUT / "r4_phase1_passive_lp_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("Wrote passive LP markouts + M67-by-day stratify; passive seller cells", len(ps))


if __name__ == "__main__":
    main()
