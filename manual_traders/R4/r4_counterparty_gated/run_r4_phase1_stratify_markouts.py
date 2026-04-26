"""
Phase-1 supplement: stratify participant markouts (aggressive buy / aggressive sell) by
ts_bucket, spread_bucket, and burst_ge4 from r4_phase1_trade_events.csv.

Also: participant signed volume and net notional (buyer +|p*q|, seller -|p*q|) from trade tape.

Output:
  r4_stratified_markout_mark67_extract_K5.csv (Mark 67 extract aggr buy — template row)
  r4_stratified_markout_all_significant.csv (cells n>=20, |t|>=2 on mean fwd for K=5,20,100)
  r4_participant_net_flow_r4.csv
  r4_baseline_resid_pos_tail.csv (largest mean residual by pair×symbol, n>=15 in cell)

Rerun: python3 run_r4_phase1_stratify_markouts.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
EV = Path(__file__).resolve().parent / "analysis_outputs" / "r4_phase1_trade_events.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
HORIZONS = (5, 20, 100)
N_MIN_CELL = 10
N_MIN_BOOT = 20


def safe_tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    if s < 1e-12:
        return float("nan")
    return m / (s / math.sqrt(n))


def boot_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < N_MIN_BOOT:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(EV, sep=",")

    def rows_for_strata(sub: pd.DataFrame, kcol: str) -> dict:
        if sub.empty or kcol not in sub.columns:
            return {}
        x = sub[kcol].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < N_MIN_CELL:
            return {
                "n": n,
                "mean": float(np.mean(x)) if n else float("nan"),
                "median": float(np.median(x)) if n else float("nan"),
                "t_stat": float("nan"),
                "frac_pos": float(np.mean(x > 0)) if n else float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
            }
        t = safe_tstat(x)
        lo, hi = boot_mean_ci(x, seed=42)
        return {
            "n": n,
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "t_stat": t,
            "frac_pos": float(np.mean(x > 0)),
            "ci_lo": lo,
            "ci_hi": hi,
        }

    # long format: one row per (participant, role, stratum) with stats for K=5,20,100
    recs: list[dict] = []
    for ag, role_tag in [("aggr_buy", "U_aggr_buy"), ("aggr_sell", "U_aggr_sell")]:
        sub_e = ev[ev["aggression"] == ag]
        if ag == "aggr_buy":
            sub_e = sub_e.copy()
            sub_e["U"] = sub_e["buyer"].astype(str)
        else:
            sub_e = sub_e.copy()
            sub_e["U"] = sub_e["seller"].astype(str)
        for U, g in sub_e.groupby("U"):
            if not U or U == "nan":
                continue
            for sym, g2 in g.groupby("symbol"):
                for tsb in sorted(g2["ts_bucket"].dropna().unique()):
                    for spr in sorted(g2["spread_bucket"].dropna().unique(), key=str):
                        for br in [0, 1]:
                            g3 = g2[
                                (g2["ts_bucket"] == tsb)
                                & (g2["spread_bucket"] == spr)
                                & (g2["burst_ge4"] == br)
                            ]
                            if len(g3) < N_MIN_CELL:
                                continue
                            for K in HORIZONS:
                                kcol = f"fwd_mid_{K}"
                                st = rows_for_strata(g3, kcol)
                                if not st:
                                    continue
                                recs.append(
                                    {
                                        "participant": U,
                                        "role": role_tag,
                                        "symbol": sym,
                                        "K": K,
                                        "ts_bucket": int(tsb) if pd.notna(tsb) else -1,
                                        "spread_bucket": str(spr),
                                        "burst_ge4": int(br),
                                        **st,
                                    }
                                )
    strat = pd.DataFrame(recs)
    if not strat.empty:
        strat.to_csv(OUT / "r4_stratified_markout_by_session_spread_burst.csv", index=False)
        m67 = strat[
            (strat["participant"] == "Mark 67")
            & (strat["symbol"] == "VELVETFRUIT_EXTRACT")
            & (strat["role"] == "U_aggr_buy")
        ]
        m67.to_csv(OUT / "r4_stratified_markout_mark67_extract_tight_allK.csv", index=False)
        sig = strat[
            (strat["n"] >= 20)
            & (strat["t_stat"].notna())
            & (strat["t_stat"].abs() >= 2.0)
        ].sort_values("t_stat", key=abs, ascending=False)
        sig.head(5000).to_csv(OUT / "r4_stratified_markout_cells_n20_t2.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT / "r4_stratified_markout_by_session_spread_burst.csv", index=False)

    # net flow from trades
    tr = pd.concat(
        [
            pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d)
            for d in (1, 2, 3)
        ],
        ignore_index=True,
    )
    tr["p"] = pd.to_numeric(tr["price"], errors="coerce")
    tr["q"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0)
    tr["gross"] = (tr["p"] * tr["q"]).abs()
    buy = (
        tr.groupby("buyer")
        .agg(
            n_buy_trades=("symbol", "size"),
            vol_buy_shares=("q", "sum"),
            gross_notional_buy=("gross", "sum"),
        )
        .reset_index()
        .rename(columns={"buyer": "name"})
    )
    sell = (
        tr.groupby("seller")
        .agg(
            n_sell_trades=("symbol", "size"),
            vol_sell_shares=("q", "sum"),
            gross_notional_sell=("gross", "sum"),
        )
        .reset_index()
        .rename(columns={"seller": "name"})
    )
    fl = buy.merge(sell, on="name", how="outer").fillna(0)
    fl["net_gross_imbalance"] = fl["gross_notional_buy"] - fl["gross_notional_sell"]
    fl["total_gross_n"] = fl["gross_notional_buy"] + fl["gross_notional_sell"]
    fl = fl.sort_values("total_gross_n", ascending=False)
    fl.to_csv(OUT / "r4_participant_net_flow_r4.csv", index=False)

    # residual positive tail: cell-level
    rpath = OUT / "r4_trade_level_residual_fwd20.csv"
    if rpath.is_file():
        r = pd.read_csv(rpath)
        cell = (
            r.groupby(["buyer", "seller", "symbol"], as_index=False)
            .agg(mean_resid=("resid_fwd20", "mean"), n_cell=("resid_fwd20", "count"))
        )
        cell = cell[cell["n_cell"] >= 20]
        cell = cell.sort_values("mean_resid", ascending=False).head(40)
        cell.to_csv(OUT / "r4_baseline_resid_pos_tail_n20.csv", index=False)
    n_strat = 0
    if not strat.empty:
        n_strat = int(len(strat))
    (OUT / "r4_stratify_markouts_summary.json").write_text(
        json.dumps(
            {
                "n_stratified_cells_written": n_strat,
                "min_n_per_cell": N_MIN_CELL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote stratified + net flow; strat rows =", n_strat)


if __name__ == "__main__":
    main()
