#!/usr/bin/env python3
"""
Phase 1 supplement — **baseline residuals (cross-asset)** and **passive adverse selection**

**Bullet 2 extension:** **Coarse** baseline = mean **fwd_EXTRACT_20** by ``(symbol, spread_bin)``
only (expected cross-asset move for that product and spread regime, ignoring counterparty).
Per-print residual = actual − baseline; aggregate by ``(buyer, seller, symbol)`` with
per-day means (detects pairs whose prints coincide with extract drift **beyond** the
symbol×spread average — not tautological like residualizing within the same cell as the pair).

**Bullet 5 extension:** On each print, identify the **passive** counterparty (non-aggressor).
  - ``aggr_buy``: taker bought at ask → **passive seller**; adverse proxy = ``fwd_same_20``
    (mid rise after you sold = bad).
  - ``aggr_sell``: taker sold at bid → **passive buyer**; adverse proxy = ``-fwd_same_20``
    (mid fall after you bought = bad).

Writes under ``outputs/``:
  - ``phase16_baseline_fwd_extract_residual_by_pair.csv``
  - ``phase16_baseline_fwd_extract_residual_summary.txt``
  - ``phase16_passive_adverse_by_party_symbol.csv``
  - ``phase16_passive_adverse_summary.txt``

Run:
  python3 manual_traders/R4/r4_phase1_marks/analyze_r4_phase1_baseline_passive_supplement.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = [1, 2, 3]
MIN_CELL = 5
MIN_AGG = 25
K = 20
COL_EXT = f"fwd_EXTRACT_{K}"
COL_SAME = f"fwd_same_{K}"


def load_te():
    spec = importlib.util.spec_from_file_location("p1", HERE / "analyze_phase1.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod.build_trade_enriched()


def baseline_extract_residuals(te: pd.DataFrame) -> pd.DataFrame:
    te = te.copy()
    grp = te.groupby(["symbol", "spread_bin"], observed=True)
    cnt = grp[COL_EXT].transform("count")
    cell_mean = grp[COL_EXT].transform("mean")
    te["residual_fwd_extract_20"] = np.where(
        cnt >= MIN_CELL,
        te[COL_EXT].astype(float) - cell_mean.astype(float),
        np.nan,
    )
    valid = te[np.isfinite(te["residual_fwd_extract_20"])].copy()
    rows = []
    for (buyer, seller, sym), g in valid.groupby(["buyer", "seller", "symbol"]):
        if len(g) < MIN_AGG:
            continue
        rvals = g["residual_fwd_extract_20"].astype(float).values
        by_day = {d: float(g.loc[g["day"] == d, "residual_fwd_extract_20"].mean()) for d in DAYS if (g["day"] == d).any()}
        signs = [np.sign(by_day[d]) for d in sorted(by_day) if np.isfinite(by_day[d]) and by_day[d] != 0]
        same_sign = len(set(signs)) <= 1 if signs else False
        rows.append(
            {
                "buyer": buyer,
                "seller": seller,
                "symbol": sym,
                "n": len(g),
                "mean_residual_ext": float(np.mean(rvals)),
                "std_residual_ext": float(np.std(rvals, ddof=1)) if len(rvals) > 1 else float("nan"),
                "mean_res_day1": by_day.get(1, float("nan")),
                "mean_res_day2": by_day.get(2, float("nan")),
                "mean_res_day3": by_day.get(3, float("nan")),
                "days_same_sign_nonzero": bool(same_sign),
            }
        )
    df = pd.DataFrame(rows).sort_values("mean_residual_ext", key=np.abs, ascending=False)
    df.to_csv(OUT / "phase16_baseline_fwd_extract_residual_by_pair.csv", index=False)
    top = df.head(25)
    lines = [
        "Residual = fwd_EXTRACT_20(print) - mean(fwd_EXTRACT_20) in same (symbol, spread_bin) pool (n_pool>=5).\n",
        f"Pooled rows with residual: {len(valid)} / {len(te)}\n\n",
        "Top |mean_residual| among (buyer,seller,symbol) with n>=25:\n",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"  {r['buyer']}->{r['seller']} {r['symbol']} n={int(r['n'])} "
            f"mean_res={r['mean_residual_ext']:.5g} d1={r['mean_res_day1']:.4g} d2={r['mean_res_day2']:.4g} d3={r['mean_res_day3']:.4g} "
            f"same_sign={r['days_same_sign_nonzero']}\n"
        )
    (OUT / "phase16_baseline_fwd_extract_residual_summary.txt").write_text("".join(lines), encoding="utf-8")
    return df


def passive_adverse(te: pd.DataFrame) -> pd.DataFrame:
    """Sign-corrected adverse proxy for the passive party on each print."""
    te = te.copy()
    buy = te[te["side"] == "aggr_buy"].copy()
    buy["passive_party"] = buy["seller"].astype(str)
    buy["adverse_proxy"] = buy[COL_SAME].astype(float)

    sell = te[te["side"] == "aggr_sell"].copy()
    sell["passive_party"] = sell["buyer"].astype(str)
    sell["adverse_proxy"] = -sell[COL_SAME].astype(float)

    comb = pd.concat(
        [
            buy[["day", "symbol", "passive_party", "adverse_proxy", "buyer", "seller"]],
            sell[["day", "symbol", "passive_party", "adverse_proxy", "buyer", "seller"]],
        ],
        ignore_index=True,
    )
    comb = comb[np.isfinite(comb["adverse_proxy"].astype(float))]
    rows = []
    for (party, sym), g in comb.groupby(["passive_party", "symbol"]):
        x = g["adverse_proxy"].astype(float).values
        if len(x) < MIN_AGG:
            continue
        by_d = {d: float(g.loc[g["day"] == d, "adverse_proxy"].mean()) for d in DAYS if (g["day"] == d).any()}
        rows.append(
            {
                "passive_party": party,
                "symbol": sym,
                "n": len(x),
                "mean_adverse": float(np.mean(x)),
                "median_adverse": float(np.median(x)),
                "frac_adverse_pos": float(np.mean(x > 0)),
                "mean_d1": by_d.get(1, float("nan")),
                "mean_d2": by_d.get(2, float("nan")),
                "mean_d3": by_d.get(3, float("nan")),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_adverse", ascending=False)
    df.to_csv(OUT / "phase16_passive_adverse_by_party_symbol.csv", index=False)

    ext = df[df["symbol"] == "VELVETFRUIT_EXTRACT"].sort_values("mean_adverse", ascending=False)
    lines = [
        "Passive adverse proxy at K=20 price bars (see module docstring for sign).\n",
        "Higher mean_adverse = worse for passive liquidity on that side.\n\n",
        "VELVETFRUIT_EXTRACT — all passive parties with n>=25 (sorted worst first):\n",
    ]
    for _, r in ext.iterrows():
        lines.append(
            f"  {r['passive_party']} n={int(r['n'])} mean={r['mean_adverse']:.5g} "
            f"frac>0={r['frac_adverse_pos']:.3f} d1={r['mean_d1']:.4g} d2={r['mean_d2']:.4g} d3={r['mean_d3']:.4g}\n"
        )
    lines.append("\nWorst 15 any symbol:\n")
    for _, r in df.head(15).iterrows():
        lines.append(
            f"  {r['passive_party']} {r['symbol']} n={int(r['n'])} mean={r['mean_adverse']:.5g}\n"
        )
    (OUT / "phase16_passive_adverse_summary.txt").write_text("".join(lines), encoding="utf-8")
    return df


def main() -> None:
    te = load_te()
    baseline_extract_residuals(te)
    passive_adverse(te)
    print("Wrote phase16_* baseline extract residuals + passive adverse tables")


if __name__ == "__main__":
    main()
