#!/usr/bin/env python3
"""
Phase 1 bullet 1 — **bootstrap 95% CI** for participant markouts **× burst_bucket**.

Uses the same **burst_bucket** convention as ``analyze_r4_phase1_participant_burst_stratify.py``:
``burst`` (n_prints_ts ≥ 4), ``isolated`` (1), ``small_multi`` (2–3).

For each cell (**participant**, **side**, **symbol**, **burst_bucket**, **K**) with
``n >= MIN_N``, bootstrap **2000** resamples of the **mean** of ``fwd_same_K`` and
``fwd_EXTRACT_K`` (paired rows where both finite).

**Horizon K:** unique timestamp bar steps (``analyze_phase1``).

Outputs:
  - ``phase18_participant_burst_bootstrap.csv``
  - ``phase18_participant_burst_bootstrap_summary.txt``

Run:
  python3 manual_traders/R4/r4_phase1_marks/analyze_r4_phase1_participant_burst_bootstrap.py
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
KS = (5, 20, 100)
MIN_N = 30
N_BOOT = 2000
RNG = np.random.default_rng(2027)


def load_p1():
    spec = importlib.util.spec_from_file_location("p1", HERE / "analyze_phase1.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def attach_burst_bucket(te: pd.DataFrame, tr_raw: pd.DataFrame) -> pd.DataFrame:
    cnt = tr_raw.groupby(["day", "timestamp"]).size().reset_index(name="n_prints_ts")
    m = te.merge(cnt, on=["day", "timestamp"], how="left")
    n = m["n_prints_ts"].fillna(1).astype(int)
    m["burst_bucket"] = np.where(
        n >= 4,
        "burst",
        np.where(n == 1, "isolated", np.where((n >= 2) & (n <= 3), "small_multi", "other")),
    )
    return m


def bootstrap_mean_ci(x: np.ndarray):
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"), float("nan"))
    means = np.empty(N_BOOT, dtype=float)
    n = len(x)
    for b in range(N_BOOT):
        s = RNG.choice(x, size=n, replace=True)
        means[b] = float(np.mean(s))
    lo, hi = np.percentile(means, [2.5, 97.5])
    mu = float(np.mean(x))
    return (mu, float(lo), float(hi))


def per_day_means(g: pd.DataFrame, col: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for d in DAYS:
        sub = g[g["day"] == d][col].astype(float)
        sub = sub[np.isfinite(sub)]
        out[d] = float(sub.mean()) if len(sub) else float("nan")
    return out


def same_sign_nonzero(means: dict[int, float]) -> bool:
    vals = [means[d] for d in DAYS if np.isfinite(means[d]) and means[d] != 0.0]
    if not vals:
        return False
    s0 = np.sign(vals[0])
    return all(np.sign(v) == s0 for v in vals)


def main() -> None:
    p1 = load_p1()
    tr_raw = pd.concat(
        [
            pd.read_csv(p1.DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d)
            for d in DAYS
        ],
        ignore_index=True,
    )
    te = p1.build_trade_enriched()
    m = attach_burst_bucket(te, tr_raw)
    m = m[m["side"].isin(("aggr_buy", "aggr_sell"))].copy()
    m["participant"] = np.where(m["side"] == "aggr_buy", m["buyer"].astype(str), m["seller"].astype(str))
    m = m[m["burst_bucket"] != "other"]

    rows = []
    for (part, side, sym, bkt), g0 in m.groupby(["participant", "side", "symbol", "burst_bucket"]):
        for K in KS:
            col = f"fwd_same_{K}"
            cex = f"fwd_EXTRACT_{K}"
            mask = g0[col].astype(float).notna() & g0[cex].astype(float).notna()
            g = g0.loc[mask]
            if len(g) < MIN_N:
                continue
            xs = g[col].astype(float).values
            xe = g[cex].astype(float).values
            mn_s, lo_s, hi_s = bootstrap_mean_ci(xs)
            mn_e, lo_e, hi_e = bootstrap_mean_ci(xe)
            dm = per_day_means(g, col)
            rows.append(
                {
                    "participant": part,
                    "side": side,
                    "symbol": sym,
                    "burst_bucket": bkt,
                    "K": K,
                    "n": len(g),
                    "mean_fwd_same": mn_s,
                    "ci95_lo_same": lo_s,
                    "ci95_hi_same": hi_s,
                    "mean_fwd_extract": mn_e,
                    "ci95_lo_extract": lo_e,
                    "ci95_hi_extract": hi_e,
                    "sign_stable_same_3d": same_sign_nonzero(dm),
                    "mean_same_d1": dm.get(1, float("nan")),
                    "mean_same_d2": dm.get(2, float("nan")),
                    "mean_same_d3": dm.get(3, float("nan")),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase18_participant_burst_bootstrap.csv", index=False)

    pos = df[(df["ci95_lo_same"] > 0) & (df["n"] >= MIN_N)]
    neg = df[(df["ci95_hi_same"] < 0) & (df["n"] >= MIN_N)]
    lines = [
        f"Bootstrap N={N_BOOT}, MIN_N={MIN_N}, seed=2027\n",
        f"Total cells: {len(df)}\n",
        f"Cells with ci95_lo(fwd_same) > 0: {len(pos)}\n",
        f"Cells with ci95_hi(fwd_same) < 0: {len(neg)}\n\n",
        "Top 15 by mean_fwd_same among cells with ci95_lo_same > 0:\n",
    ]
    if len(pos):
        for _, r in pos.sort_values("mean_fwd_same", ascending=False).head(15).iterrows():
            lines.append(
                f"  {r['participant']} {r['side']} {r['symbol']} {r['burst_bucket']} K={int(r['K'])} "
                f"n={int(r['n'])} mean={r['mean_fwd_same']:.4g} CI=[{r['ci95_lo_same']:.4g},{r['ci95_hi_same']:.4g}] "
                f"stable={r['sign_stable_same_3d']}\n"
            )
    else:
        lines.append("  (none)\n")
    lines.append("\nExtract cells (symbol VELVETFRUIT_EXTRACT) small_multi K=20, ci fully above 0:\n")
    ex = df[
        (df["symbol"] == "VELVETFRUIT_EXTRACT")
        & (df["burst_bucket"] == "small_multi")
        & (df["K"] == 20)
        & (df["ci95_lo_same"] > 0)
    ]
    if len(ex) == 0:
        lines.append("  (none)\n")
    else:
        lines.append(ex.to_string(index=False) + "\n")
    (OUT / "phase18_participant_burst_bootstrap_summary.txt").write_text("".join(lines), encoding="utf-8")
    print("Wrote phase18_* cells=", len(df), "ci_lo>0 same=", len(pos))


if __name__ == "__main__":
    main()
