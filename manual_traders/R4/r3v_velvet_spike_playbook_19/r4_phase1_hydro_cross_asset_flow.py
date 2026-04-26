#!/usr/bin/env python3
"""
Round 4 Phase 1 — HYDROGEL_PACK focus (explicit cross-asset + roles):

1) Aggressor prints on HYDROGEL_PACK only: forward **extract** mid change (fwd_K_ex)
   and forward **hydro same-symbol** mid (fwd_K_sym), by day / aggressor / side / spr_b.
   Min n per cell = 25.

2) Signed flow balance per Mark name on hydro: for each trade row on hydro,
   +quantity to buyer, -quantity to seller; aggregate by (day, name).

Inputs: r4_trades_with_markout.csv (from r4_phase1_counterparty_study.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

INP = Path(__file__).resolve().parent / "analysis_outputs" / "r4_trades_with_markout.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs"
HYDRO = "HYDROGEL_PACK"
KS = (5, 20, 100)
MIN_N = 25


def t_one(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_1samp(x, 0.0, nan_policy="omit")
    return float(t), float(p)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(INP)
    h = m[(m["product"] == HYDRO) & (m["aggr"].isin(["buy_aggr", "sell_aggr"]))].copy()

    rows = []
    for k in KS:
        c_sym = f"fwd_{k}_sym"
        c_ex = f"fwd_{k}_ex"
        for (day, party, side, sprb), g in h.groupby(["day", "agg_party", "agg_role", "spr_b"]):
            for col, label in [(c_sym, "hydro_mid"), (c_ex, "extract_mid")]:
                x = g[col].dropna().to_numpy(dtype=float)
                n = len(x)
                if n < MIN_N:
                    continue
                tm, pv = t_one(x)
                rows.append(
                    {
                        "K": k,
                        "forward_on": label,
                        "day": int(day),
                        "aggressor_party": str(party),
                        "aggressor_side": str(side),
                        "spread_bucket": str(sprb),
                        "n": n,
                        "mean": float(x.mean()),
                        "median": float(np.median(x)),
                        "frac_pos": float((x > 0).mean()),
                        "tstat_vs0": tm,
                        "pvalue_vs0": pv,
                    }
                )
    df_m = pd.DataFrame(rows).sort_values(["K", "forward_on", "day", "n"], ascending=[True, True, True, False])
    df_m.to_csv(OUT / "r4_phase1_hydro_aggressor_fwd_cross_asset_by_day.csv", index=False)

    # Signed flow on hydro from raw tape (all trades, not only aggressor)
    DATA = Path(__file__).resolve().parents[3] / "Prosperity4Data" / "ROUND_4"
    flow_rows = []
    for d in (1, 2, 3):
        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        tr = tr[tr["symbol"] == HYDRO].copy()
        tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0)
        for name in pd.unique(pd.concat([tr["buyer"].astype(str), tr["seller"].astype(str)])):
            if not name or name == "nan":
                continue
            buy_q = tr.loc[tr["buyer"].astype(str) == name, "quantity"].sum()
            sell_q = tr.loc[tr["seller"].astype(str) == name, "quantity"].sum()
            flow_rows.append(
                {
                    "day": d,
                    "name": name,
                    "buy_qty": float(buy_q),
                    "sell_qty": float(sell_q),
                    "signed_flow_buy_minus_sell": float(buy_q - sell_q),
                    "n_trades_touching": int(
                        (tr["buyer"].astype(str) == name).sum() + (tr["seller"].astype(str) == name).sum()
                    ),
                }
            )
    df_f = pd.DataFrame(flow_rows)
    parts = []
    for d in (1, 2, 3):
        sub = df_f[df_f["day"] == d].copy()
        sub["_abs"] = sub["signed_flow_buy_minus_sell"].abs()
        parts.append(sub.sort_values("_abs", ascending=False).drop(columns="_abs"))
    df_f = pd.concat(parts, ignore_index=True)
    df_f.to_csv(OUT / "r4_phase1_hydro_signed_flow_by_name_day.csv", index=False)

    summary = {
        "hydro_aggressor_cross_asset_csv": str(OUT / "r4_phase1_hydro_aggressor_fwd_cross_asset_by_day.csv"),
        "hydro_signed_flow_csv": str(OUT / "r4_phase1_hydro_signed_flow_by_name_day.csv"),
        "min_n_cells": MIN_N,
        "note": "Cross-asset columns use extract mid forward from **hydro trade timestamp** (same clock as Phase 1 markouts).",
    }
    (OUT / "r4_phase1_hydro_flow_index.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", len(df_m), "markout cells,", len(df_f), "flow rows")


if __name__ == "__main__":
    main()
