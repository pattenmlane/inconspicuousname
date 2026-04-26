#!/usr/bin/env python3
"""
Phase 1 **supplement** (Round 4 days 1–3): items from the operations template
not fully in the original `participant_markout` CSV (bootstrap CI, per-day
means, net aggressor flow, pair reciprocity, markout × burst for K=20).

- **K:** same as `analyze_phase1` (price-bar / unique `timestamp` index steps).

1) **phase1_participant_net_aggr_flow_lots.csv** — for each Mark: sum of **qty** on
   **aggr_buy** when buyer==U, minus sum on **aggr_sell** when seller==U (enriched
   trades with aggressor from Phase 1).

2) **phase1_pair_reciprocity_all_marks.csv** — n(A→B), n(B→A), reciprocity
   R = 2*min(n_ab,n_ba)/(n_ab+n_ba) for the seven named Marks (full trade tape).

3) **phase1_participant_markout_bootstrap_n50plus.csv** — same (U, aggr side, sym, K)
   cells as `participant_markout` with **n≥50**; bootstrap **mean** of
   `fwd_same_K` (2000 resamples); **day1/2/3** mean; **sign_stable_3d** (all finite
   daily means same sign or zero).

4) **phase1_participant_markout_x_burst_k20.csv** — for K=**20** and cells with
   n≥30 on full cell: split by **burst_ge4** (same (day, timestamp) ≥4 trade rows)
   mean `fwd_same_20` and n.

5) **phase1_bootstrap_summary.txt** — count cells with 95% CI lower bound > 0, etc.

Run: python3 manual_traders/R4/r4_phase1_marks/analyze_r4_phase1_stability_supplement.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# .../manual_traders/R4/r4_phase1_marks -> repo root is parents[2] == /workspace
REPO = HERE.parents[2]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = [1, 2, 3]
KS = (5, 20, 100)
N_BOOT = 2000
RNG = np.random.default_rng(2026)


def load_p1():
    spec = importlib.util.spec_from_file_location("p1", HERE / "analyze_phase1.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def net_aggr_flow_lots(te: pd.DataFrame) -> None:
    names = sorted(set(te["buyer"].astype(str)) | set(te["seller"].astype(str)))
    rows = []
    for u in names:
        buy = te[(te["side"] == "aggr_buy") & (te["buyer"] == u)]
        sell = te[(te["side"] == "aggr_sell") & (te["seller"] == u)]
        qb = int(buy["qty"].sum()) if len(buy) else 0
        qs = int(sell["qty"].sum()) if len(sell) else 0
        rows.append(
            {
                "mark": u,
                "n_aggr_buy": len(buy),
                "n_aggr_sell": len(sell),
                "lots_aggr_buy": qb,
                "lots_aggr_sell": qs,
                "net_aggr_lots_buy_minus_sell": qb - qs,
            }
        )
    pd.DataFrame(rows).sort_values("mark").to_csv(OUT / "phase1_participant_net_aggr_flow_lots.csv", index=False)
    print("Wrote", OUT / "phase1_participant_net_aggr_flow_lots.csv")


def pair_reciprocity() -> None:
    tr = pd.concat(
        [pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";") for d in DAYS],
        ignore_index=True,
    )
    marks = sorted(set(tr["buyer"].astype(str)) | set(tr["seller"].astype(str)))
    rrows = []
    for a in marks:
        for b in marks:
            if a == b:
                continue
            n_ab = int(((tr["buyer"] == a) & (tr["seller"] == b)).sum())
            n_ba = int(((tr["buyer"] == b) & (tr["seller"] == a)).sum())
            t = n_ab + n_ba
            r = 2.0 * min(n_ab, n_ba) / t if t else float("nan")
            rrows.append(
                {
                    "buyer": a,
                    "seller": b,
                    "n_ab": n_ab,
                    "n_ba": n_ba,
                    "reciprocity_R": r,
                }
            )
    rdf = pd.DataFrame(rrows).sort_values("n_ab", ascending=False)
    rdf.to_csv(OUT / "phase1_pair_reciprocity_all_marks.csv", index=False)
    print("Wrote", OUT / "phase1_pair_reciprocity_all_marks.csv")


def main() -> None:
    p1 = load_p1()
    tr_raw = pd.concat(
        [pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS],
        ignore_index=True,
    )
    te = p1.build_trade_enriched()
    bf = p1.burst_flags(tr_raw)
    burst_map = bf.set_index(["day", "timestamp"])["burst_ge4"].to_dict()
    te = te.copy()
    te["burst_ge4"] = te.apply(
        lambda r: burst_map.get((int(r["day"]), int(r["timestamp"])), False), axis=1
    )

    net_aggr_flow_lots(te)
    pair_reciprocity()

    boot_rows = []
    burst_rows = []
    names = sorted(set(te["buyer"].astype(str)) | set(te["seller"].astype(str)))
    for u in names:
        buy_u = te[(te["side"] == "aggr_buy") & (te["buyer"] == u)]
        sell_u = te[(te["side"] == "aggr_sell") & (te["seller"] == u)]
        for side, sub in (("aggr_buy", buy_u), ("aggr_sell", sell_u)):
            for sym in sub["symbol"].unique():
                s2 = sub[sub["symbol"] == sym]
                if len(s2) < 20:
                    continue
                for K in KS:
                    col = f"fwd_same_{K}"
                    x = s2[col].astype(float).values
                    x = x[np.isfinite(x)]
                    if len(x) < 50:
                        continue
                    boot = [float(np.mean(RNG.choice(x, size=len(x), replace=True))) for _ in range(N_BOOT)]
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    m1 = s2[s2["day"] == 1][col].astype(float)
                    m2 = s2[s2["day"] == 2][col].astype(float)
                    m3 = s2[s2["day"] == 3][col].astype(float)
                    d1 = float(m1.mean()) if m1.notna().any() else float("nan")
                    d2 = float(m2.mean()) if m2.notna().any() else float("nan")
                    d3 = float(m3.mean()) if m3.notna().any() else float("nan")
                    day_vals = [v for v in (d1, d2, d3) if np.isfinite(v)]
                    nz = [v for v in day_vals if v != 0.0]
                    if not nz:
                        stable = 1
                    elif all(v > 0 for v in nz) or all(v < 0 for v in nz):
                        stable = 1
                    else:
                        stable = 0

                    boot_rows.append(
                        {
                            "mark": u,
                            "side": side,
                            "symbol": sym,
                            "K": K,
                            "n": len(x),
                            "mean_fwd_same": float(np.mean(x)),
                            "ci95_lo": float(lo),
                            "ci95_hi": float(hi),
                            "day1_mean": d1,
                            "day2_mean": d2,
                            "day3_mean": d3,
                            "sign_strict_same_sign_3d": int(stable),
                        }
                    )
                    if K == 20 and len(s2) >= 30:
                        for bflag, label in ((True, "burst_ge4"), (False, "not_burst")):
                            g = s2[s2["burst_ge4"] == bflag]
                            y = g[col].astype(float).values
                            y = y[np.isfinite(y)]
                            if len(y) < 8:
                                continue
                            burst_rows.append(
                                {
                                    "mark": u,
                                    "side": side,
                                    "symbol": sym,
                                    "bucket": label,
                                    "n": len(y),
                                    "mean_fwd_same": float(np.mean(y)),
                                }
                            )
    bdf = pd.DataFrame(boot_rows)
    bdf.to_csv(OUT / "phase1_participant_markout_bootstrap_n50plus.csv", index=False)
    pd.DataFrame(burst_rows).to_csv(OUT / "phase1_participant_markout_x_burst_k20.csv", index=False)
    n_pos = int((bdf["ci95_lo"] > 0).sum())
    n_str = int(bdf["sign_strict_same_sign_3d"].sum())
    with open(OUT / "phase1_bootstrap_summary.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Cells with n>=50: {len(bdf)}\n"
            f"95% bootstrap CI: lower bound > 0 (strict positive): {n_pos}\n"
            f" sign_strict_same_sign_3d: {n_str} / {len(bdf)}\n"
        )
    print("Wrote", OUT / "phase1_bootstrap_summary.txt", "pos_ci", n_pos)


if __name__ == "__main__":
    main()
