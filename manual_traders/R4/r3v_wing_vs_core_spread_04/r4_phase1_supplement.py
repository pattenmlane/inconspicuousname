#!/usr/bin/env python3
"""
Phase 1 supplement — official checklist items not summarized in a single CSV:

- Bootstrap 95% CI (percentile) for **pooled** mean fwd_same on participant_markout_long groups
  with n >= 20 (B=5000, seed=0). Written for k in {5, 20, 100} to separate CSVs.
- **Role / flow balance** per name: count rows as buyer vs seller, net signed quantity
  (buy quantity − sell quantity), and top directed pairs involving that name.

Requires outputs from r4_phase1_counterparty_analysis.py (participant_markout_long.csv, trades in DATA).

Run from repo root:
  python3 manual_traders/R4/r3v_wing_vs_core_spread_04/r4_phase1_supplement.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = Path(__file__).resolve().parent / "outputs" / "phase1"
DAYS = [1, 2, 3]
BOOT_B = 5000
BOOT_SEED = 0
MIN_N = 20
KS = (5, 20, 100)


def bootstrap_mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float(np.nan), float(np.nan), float(np.nan)
    mean = float(np.mean(x))
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(BOOT_B, n))
    means = x[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return mean, lo, hi


def main() -> None:
    long_path = OUT / "participant_markout_long.csv"
    if not long_path.is_file():
        raise SystemExit(f"Missing {long_path}; run r4_phase1_counterparty_analysis.py first")

    part = pd.read_csv(long_path)
    gcols = ["name", "role", "symbol", "k", "spread_regime", "burst"]

    for k in KS:
        sub = part[part["k"] == k]
        rows = []
        for key, g in sub.groupby(gcols, dropna=False):
            y = g["fwd_same"].to_numpy(dtype=float)
            y = y[np.isfinite(y)]
            n = len(y)
            if n < MIN_N:
                continue
            m, lo, hi = bootstrap_mean_ci(y)
            d = dict(zip(gcols, key, strict=True))
            d["n"] = n
            d["mean_fwd_same"] = m
            d["ci95_lo"] = lo
            d["ci95_hi"] = hi
            d["t_stat"] = m / (float(np.std(y, ddof=1)) / np.sqrt(n)) if n > 1 and np.std(y, ddof=1) > 1e-12 else float("nan")
            d["frac_pos"] = float(np.mean(y > 0))
            rows.append(d)
        out = pd.DataFrame(rows)
        if len(out):
            out = out.sort_values(["k", "t_stat"], ascending=[True, False], na_position="last")
        out.to_csv(OUT / f"participant_pooled_bootstrap_k{k}.csv", index=False)
        print("Wrote", OUT / f"participant_pooled_bootstrap_k{k}.csv", "rows", len(out))

    # Per-name flow / imbalance (all products)
    frames = []
    for d in DAYS:
        p = DATA / f"trades_round_4_day_{d}.csv"
        if p.is_file():
            t = pd.read_csv(p, sep=";")
            t["tape_day"] = d
            frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    tr["quantity"] = pd.to_numeric(tr["quantity"], errors="coerce").fillna(0).astype(int)
    names = sorted(set(tr["buyer"].astype(str)) | set(tr["seller"].astype(str)))
    flow_rows = []
    for nm in names:
        as_b = tr[tr["buyer"] == nm]
        as_s = tr[tr["seller"] == nm]
        n_buy = int(len(as_b))
        n_sell = int(len(as_s))
        q_buy = int(as_b["quantity"].abs().sum())
        q_sell = int(as_s["quantity"].abs().sum())
        flow_rows.append(
            {
                "name": nm,
                "n_prints_as_buyer": n_buy,
                "n_prints_as_seller": n_sell,
                "qty_bought": q_buy,
                "qty_sold": q_sell,
                "net_qty_signed": q_buy - q_sell,
            }
        )
    flow_df = pd.DataFrame(flow_rows).sort_values("n_prints_as_buyer", ascending=False)
    flow_df.to_csv(OUT / "participant_flow_balance.csv", index=False)
    print("Wrote", OUT / "participant_flow_balance.csv")

    bullet_index = f"""Round 4 Phase 1 — output index (suggested direction.txt)

Horizon K: K consecutive mid_price tape rows (100 time units per step on the 10k grid).

Bullet 1 — Participant markouts: participant_markout_long.csv (per row), participant_markout_pooled.csv
  (pooled t-stat), participant_markout_by_day_nosession.csv (stability by tape_day),
  participant_pooled_bootstrap_k{{5,20,100}}.csv (bootstrap 95% CI for pooled means, n>={MIN_N}).

Bullet 2 — Baseline: per_print_with_baseline.csv, pair_baseline_residuals.csv. Flow / roles: participant_flow_balance.csv

Bullet 3 — Graph: graph_edges.csv, graph_top_pairs.txt, graph_hubs.json; twohop_chain_counts.csv, twohop_chain_fwd_extract.csv

Bullet 4 — Bursts: burst_events.csv, burst_forward_extract.csv (burst vs control)

Bullet 5 — Passive adverse: passive_adverse_by_pair.csv

Summary: phase1_summary.json
Script: r4_phase1_counterparty_analysis.py, r4_phase1_supplement.py
"""
    (OUT / "phase1_bullet_file_index.txt").write_text(bullet_index, encoding="utf-8")
    print("Wrote", OUT / "phase1_bullet_file_index.txt")

    # Top Mark 67 buy_aggr + extract + tight + burst=0 + k=20 for doc
    p20 = OUT / "participant_pooled_bootstrap_k20.csv"
    if p20.is_file():
        t = pd.read_csv(p20)
        mask = (t["name"] == "Mark 67") & (t["role"] == "buy_aggr") & (t["symbol"] == "VELVETFRUIT_EXTRACT")
        t = t[mask & (t["spread_regime"] == "tight") & (t["burst"] == 0)]
        if len(t):
            r = t.iloc[0].to_dict()
        else:
            r = {"note": "no row; relax filters in CSV"}
        (OUT / "phase1_bootstrap_m67_extract_tight_k20.json").write_text(
            json.dumps(r, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
