#!/usr/bin/env python3
"""
Phase 1 — burst event study with **session-matched random controls** (non-burst prints).

- Burst: (day, timestamp) with >= 4 trade rows (same as Phase 1).
- Control pool: (day, timestamp) with exactly 1 trade row, same `session` bucket as the burst row
  (session from first trade row in enriched CSV for that (day, ts)).
- For each replicate, sample N controls equal to N burst events (with replacement per replicate),
  where N = number of distinct burst (day, ts) after merging mark_20_u.
- Report mean mark_20_u for burst vs mean for random control (2000 replicates) and p-value:
  fraction of replicates with control mean >= burst mean (one-sided, tests if burst > random).

Input: r4_trades_enriched_markouts.csv
Output: r4_phase1_burst_vs_session_random.json, r4_phase1_burst_vs_session_random.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
ENR = Path(__file__).resolve().parent / "outputs" / "r4_trades_enriched_markouts.csv"
OUT = Path(__file__).resolve().parent / "outputs"
RNG = np.random.default_rng(7)
N_REP = 2000


def main() -> None:
    if not ENR.is_file():
        raise SystemExit(f"missing {ENR}")
    df = pd.read_csv(ENR)
    df["mark_20_u"] = pd.to_numeric(df["mark_20_u"], errors="coerce")

    tr_parts = []
    for d in (1, 2, 3):
        t = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        t["day"] = d
        tr_parts.append(t)
    tr = pd.concat(tr_parts, ignore_index=True)
    cnt = tr.groupby(["day", "timestamp"]).size().rename("n_prints")

    bdf = (
        pd.DataFrame({"n_prints": cnt})
        .reset_index()
        .assign(
            burst=lambda x: x["n_prints"] >= 4,
            is_single=lambda x: x["n_prints"] == 1,
        )
    )

    m = df.drop_duplicates(["day", "timestamp"])[
        ["day", "timestamp", "session", "mark_20_u"]
    ]
    bdf = bdf.merge(m, on=["day", "timestamp"], how="left")

    burst_df = bdf[bdf["burst"]].dropna(subset=["mark_20_u"])
    single_by_sess: dict[str, list[float]] = {}
    for _, r in bdf[bdf["is_single"]].iterrows():
        sess = str(r["session"])
        mu = r["mark_20_u"]
        if pd.isna(mu):
            continue
        single_by_sess.setdefault(sess, []).append(float(mu))

    burst_vals: list[tuple[str, float]] = []
    for _, r in burst_df.iterrows():
        burst_vals.append((str(r["session"]), float(r["mark_20_u"])))
    n_burst = len(burst_vals)
    mean_b = float(np.mean([v for _, v in burst_vals]))

    # per-session control pools
    null_means = []
    for _ in range(N_REP):
        picked = []
        for sess, v in burst_vals:
            pool = single_by_sess.get(sess)
            if not pool or len(pool) < 2:
                # fallback: any session pool
                pool = [x for xs in single_by_sess.values() for x in xs]
            picked.append(RNG.choice(pool))
        null_means.append(float(np.mean(picked)))

    null_arr = np.array(null_means)
    p_one_sided = float(np.mean(null_arr >= mean_b))
    p_two_sided = float(2 * min(p_one_sided, 1.0 - p_one_sided)) if 0 < p_one_sided < 1 else p_one_sided

    out = {
        "n_burst_timestamps": n_burst,
        "mean_mark20_u_burst": mean_b,
        "n_rep": N_REP,
        "control_pool": {k: len(v) for k, v in single_by_sess.items()},
        "random_control_mean_mark20_u": {
            "mean": float(null_arr.mean()),
            "p5": float(np.percentile(null_arr, 5)),
            "p50": float(np.percentile(null_arr, 50)),
            "p95": float(np.percentile(null_arr, 95)),
        },
        "p_value_one_sided_burst_gt_random": p_one_sided,
        "p_value_two_sided": p_two_sided,
        "method": "For each burst (day,ts) take mark_20_u; controls sample one isolated (n_prints==1) print per burst from same session, with replacement over replicates. If session empty, fallback to global isolated pool.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r4_phase1_burst_vs_session_random.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    txt = [
        f"Burst n (distinct day,timestamp with mark_20_u): {n_burst}",
        f"Mean U mark@20 (burst): {mean_b:.6f}",
        f"Null mean U mark@20 (session-matched single-print random, {N_REP} rep): {null_arr.mean():.6f}  [p5, p50, p95] = [{np.percentile(null_arr,5):.4f}, {np.percentile(null_arr,50):.4f}, {np.percentile(null_arr,95):.4f}]",
        f"One-sided p (null >= burst): {p_one_sided:.4f}",
        f"Two-sided p: {p_two_sided:.4f}",
    ]
    (OUT / "r4_phase1_burst_vs_session_random.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("wrote", OUT / "r4_phase1_burst_vs_session_random.json")


if __name__ == "__main__":
    main()
