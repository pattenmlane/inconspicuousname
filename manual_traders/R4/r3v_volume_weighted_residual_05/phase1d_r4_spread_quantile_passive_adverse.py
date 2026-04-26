#!/usr/bin/env python3
"""
Round 4 Phase 1d — **spread quantile stratification** + **passive-side adverse** markouts.

Reads `analysis_outputs/phase1/r4_trades_enriched.csv`.

1) **Spread quantile (per day × symbol):** assign each trade row to `spr_q` ∈ {Q1..Q4}
   by `spread_sym` rank within `(day, symbol)` (ties get same rank; pandas qcut with duplicates).

2) **Mark67 U buy_agg × spr_q:** mean/median/n/t + bootstrap CI for `fwd_u_20` and `fwd_mid_20`
   per quantile (Phase-1 “stratify by spread quantile”).

3) **Passive adverse (population):**
   - **Passive at ask (lifted):** `aggressor==buy_agg` and `seller==Mark` — Mark sold into an
     aggressive buyer (provided ask-side liquidity).
   - **Passive at bid (hit):** `aggressor==sell_agg` and `buyer==Mark` — Mark bought from an
     aggressive seller (provided bid-side liquidity).
   For each Mark and role, same-symbol fwd @ K∈{5,20} (negative mean = adverse for that passive side).

4) **Pair / net notionals:** per Mark, sum notional as buyer vs as seller (tape-level imbalance hint).

Outputs under `analysis_outputs/phase1/`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

INP = Path(__file__).resolve().parent / "analysis_outputs" / "phase1" / "r4_trades_enriched.csv"
OUT = Path(__file__).resolve().parent / "analysis_outputs" / "phase1"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(7)
BOOT = 2000


def bootstrap_mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(x))
    idx = RNG.integers(0, len(x), size=(BOOT, len(x)))
    mus = x[idx].mean(axis=1)
    return m, float(np.percentile(mus, 2.5)), float(np.percentile(mus, 97.5))


def t_stat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    se = s / math.sqrt(len(x)) if s > 0 else float("nan")
    return m / se if se and math.isfinite(se) else float("nan")


def add_spread_quantile(ev: pd.DataFrame) -> pd.DataFrame:
    def qcut_grp(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.nunique() < 2:
            return pd.Series(["Q_mid"] * len(s), index=s.index)
        try:
            return pd.qcut(s.rank(method="first"), 4, labels=["Q1_tight", "Q2", "Q3", "Q4_wide"])
        except ValueError:
            return pd.qcut(s, 4, labels=["Q1_tight", "Q2", "Q3", "Q4_wide"], duplicates="drop")

    ev = ev.copy()
    ev["spr_q"] = ev.groupby(["day", "symbol"], group_keys=False)["spread_sym"].transform(qcut_grp)
    return ev


def main() -> None:
    ev = pd.read_csv(INP)
    ev = add_spread_quantile(ev)

    # --- Mark67 U buy_agg by spread quantile ---
    m67u = ev[
        (ev["symbol"] == "VELVETFRUIT_EXTRACT")
        & (ev["buyer"] == "Mark 67")
        & (ev["aggressor"] == "buy_agg")
    ].copy()
    rows_q = []
    for q, g in m67u.groupby("spr_q"):
        for K in (5, 20):
            for col, label in [(f"fwd_u_{K}", "fwd_u"), (f"fwd_mid_{K}", "fwd_same")]:
                x = pd.to_numeric(g[col], errors="coerce").dropna().to_numpy()
                if len(x) < 5:
                    continue
                m, lo, hi = bootstrap_mean_ci(x)
                rows_q.append(
                    {
                        "slice": "Mark67|U|buy_agg",
                        "spr_q": str(q),
                        "K": K,
                        "fwd": label,
                        "n": len(x),
                        "mean": float(np.mean(x)),
                        "median": float(np.median(x)),
                        "t_stat": t_stat(x),
                        "frac_pos": float((x > 0).mean()),
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                    }
                )
    pd.DataFrame(rows_q).to_csv(OUT / "r4_phase1d_mark67_u_buyagg_by_spread_quantile.csv", index=False)

    # --- Passive adverse by Mark ---
    names = sorted(set(ev["buyer"]) | set(ev["seller"]))
    adv_rows = []
    for mark in names:
        if mark.startswith("Mark"):
            pass
        else:
            continue
        # passive ask-side (sold stock to aggressive buyer)
        g1 = ev[(ev["aggressor"] == "buy_agg") & (ev["seller"] == mark)]
        # passive bid-side (bought from aggressive seller)
        g2 = ev[(ev["aggressor"] == "sell_agg") & (ev["buyer"] == mark)]
        for role, g in [("passive_at_ask", g1), ("passive_at_bid", g2)]:
            if len(g) < 5:
                continue
            for K in (5, 20):
                c = f"fwd_mid_{K}"
                x = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy()
                if len(x) < 5:
                    continue
                m, lo, hi = bootstrap_mean_ci(x)
                adv_rows.append(
                    {
                        "Mark": mark,
                        "passive_role": role,
                        "K": K,
                        "n": len(x),
                        "mean_fwd_same_sym": float(np.mean(x)),
                        "median": float(np.median(x)),
                        "t_stat": t_stat(x),
                        "frac_pos": float((x > 0).mean()),
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                    }
                )
    pd.DataFrame(adv_rows).to_csv(OUT / "r4_phase1d_passive_side_adverse_by_mark.csv", index=False)

    # --- Net notional buy vs sell per Mark ---
    nb = []
    for mark in names:
        if not mark.startswith("Mark"):
            continue
        buy_n = float(ev.loc[ev["buyer"] == mark, "notional"].sum())
        sell_n = float(ev.loc[ev["seller"] == mark, "notional"].sum())
        nb.append(
            {
                "Mark": mark,
                "notional_as_buyer": buy_n,
                "notional_as_seller": sell_n,
                "net_buy_minus_sell": buy_n - sell_n,
            }
        )
    nbd = pd.DataFrame(nb).sort_values("net_buy_minus_sell", ascending=False)
    nbd.to_csv(OUT / "r4_phase1d_mark_net_notional.csv", index=False)

    lines = [
        "Per-(day,symbol) spread_sym quartiles: Q1_tight = narrowest 25% of prints for that product that day.",
        "Mark67 U buy_agg by spr_q: see r4_phase1d_mark67_u_buyagg_by_spread_quantile.csv",
        "Passive adverse: negative mean fwd_same_sym after passive_at_ask suggests seller got picked off (price rose after).",
        "Net notional (buyer - seller) table: r4_phase1d_mark_net_notional.csv",
    ]
    (OUT / "r4_phase1d_summary.txt").write_text("\n".join(lines) + "\n")
    print("Wrote phase1d to", OUT)


if __name__ == "__main__":
    main()
