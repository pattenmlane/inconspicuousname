#!/usr/bin/env python3
"""
Phase 1 supplement — **graph hubs** and **burst vs random-matched** controls

**1) Hub metrics (directed Mark→Mark, full tape)**
- Out-weight / in-weight = **count** and **notional** (sum price*qty) for each U.
- Hub score = in_count + out_count; same for notional.
- **Outputs:** ``phase1_graph_hub_degree_counts.csv``,
  ``phase1_graph_hub_degree_notionals.csv``,
  ``phase1_orchestrator_at_burst_rows.csv`` (per burst timestamp: n_prints, top buyers/sellers, n_syms)

**2) Burst event study with random non-burst timestamp controls (per day)**
- **Burst** set: (day, ts) with ``n_prints >= 4`` (same as ``burst_ge4``).
- **Control** pool: (day, ts) with ``n_prints < 4`` and ``n_prints >= 1``.
- For each day D: sample **min(n_burst_D, n_control_pool_D)** timestamps uniformly **without
  replacement** from the control pool, matching the burst count for that day.
- Compare **pooled** mean ``fwd_EXTRACT_20`` (one value per *print* at the sampled
  events; same as burst branch which pools all prints at burst times).
- **Welch t** (burst print distribution vs control print distribution) + means.

Also write **isolated** comparison: prints at timestamps with **exactly 1** trade row
(vs burst) for reference.

K / horizon: same as ``analyze_phase1`` (``fwd_EXTRACT_20`` in enriched trades).

Run: python3 manual_traders/R4/r4_phase1_marks/analyze_r4_phase1_graph_hubs_burst_control.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
# .../manual_traders/R4/r4_phase1_marks -> repo root is parents[2] == /workspace
REPO = HERE.parents[2]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
DAYS = [1, 2, 3]
RNG = np.random.default_rng(7)


def t_welch(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    r = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(r.statistic)


def load_p1():
    spec = importlib.util.spec_from_file_location("p1", HERE / "analyze_phase1.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def graph_hubs(tr: pd.DataFrame) -> None:
    tr = tr.copy()
    tr["notional"] = tr["price"].astype(float) * tr["quantity"].astype(int)
    e = tr.groupby(["buyer", "seller"], as_index=False).agg(
        n_prints=("symbol", "size"), notional=("notional", "sum")
    )
    marks = sorted(set(tr["buyer"].astype(str)) | set(tr["seller"].astype(str)))
    count_rows = []
    notion_rows = []
    for u in marks:
        out_n = e.loc[e["buyer"] == u, "n_prints"].sum()
        in_n = e.loc[e["seller"] == u, "n_prints"].sum()
        out_not = e.loc[e["buyer"] == u, "notional"].sum()
        in_not = e.loc[e["seller"] == u, "notional"].sum()
        count_rows.append(
            {
                "mark": u,
                "out_count": int(out_n),
                "in_count": int(in_n),
                "in_plus_out": int(in_n + out_n),
            }
        )
        notion_rows.append(
            {
                "mark": u,
                "out_notional": float(out_not),
                "in_notional": float(in_not),
                "in_plus_out_abs": float(abs(in_not) + abs(out_not)),
            }
        )
    pd.DataFrame(count_rows).sort_values("in_plus_out", ascending=False).to_csv(
        OUT / "phase1_graph_hub_degree_counts.csv", index=False
    )
    pd.DataFrame(notion_rows).sort_values("in_plus_out_abs", ascending=False).to_csv(
        OUT / "phase1_graph_hub_degree_notionals.csv", index=False
    )
    print("Wrote graph hub tables")


def orchestrator_bursts(bf: pd.DataFrame) -> None:
    b = bf[bf["burst_ge4"]].copy()
    b.to_csv(OUT / "phase1_orchestrator_at_burst_rows.csv", index=False)
    print("Wrote", OUT / "phase1_orchestrator_at_burst_rows.csv", "n=", len(b))


def _one_day_event_study(
    d: int, cnt: pd.DataFrame, te: pd.DataFrame
) -> dict | None:
    cday = cnt[cnt["day"] == d]
    burst_all = cday.loc[cday["n_prints"] >= 4, "timestamp"].astype(int).values
    ctrl_pool = cday.loc[(cday["n_prints"] < 4) & (cday["n_prints"] >= 1), "timestamp"].astype(int).values
    n_b, n_p = len(burst_all), len(ctrl_pool)
    if n_b == 0 or n_p == 0:
        return None
    k = min(n_b, n_p)
    burst_ts = (
        burst_all
        if len(burst_all) == k
        else RNG.choice(burst_all, size=k, replace=False)
    )
    if n_p >= k:
        ctrl_ts = RNG.choice(ctrl_pool, size=k, replace=False)
    else:
        ctrl_ts = RNG.choice(ctrl_pool, size=k, replace=True)

    te_d = te[te["day"] == d]
    brv = te_d[te_d["timestamp"].astype(int).isin(burst_ts)]["fwd_EXTRACT_20"].astype(float).dropna().values
    crv = te_d[te_d["timestamp"].astype(int).isin(ctrl_ts)]["fwd_EXTRACT_20"].astype(float).dropna().values
    single_ts = cday[cday["n_prints"] == 1]["timestamp"].astype(int).values
    srv = te_d[te_d["timestamp"].astype(int).isin(single_ts)]["fwd_EXTRACT_20"].astype(float).dropna().values
    t_bc = t_welch(brv, crv) if len(brv) > 1 and len(crv) > 1 else float("nan")
    return {
        "day": d,
        "k_matched": k,
        "burst_ts_sampled": bool(len(burst_all) > k),
        "n_burst_prints": len(brv),
        "mean_burst": float(np.mean(brv)) if len(brv) else float("nan"),
        "n_ctrl_prints": len(crv),
        "mean_ctrl": float(np.mean(crv)) if len(crv) else float("nan"),
        "welch_t_b_minus_c": t_bc,
        "n_isolated_prints": len(srv),
        "mean_isolated": float(np.mean(srv)) if len(srv) else float("nan"),
        "brv": brv,
        "crv": crv,
        "srv": srv,
    }


def burst_event_random_control(te: pd.DataFrame, tr_all: pd.DataFrame) -> None:
    cnt = tr_all.groupby(["day", "timestamp"]).size().reset_index(name="n_prints")
    lines = [
        "Burst vs random-matched *non-burst* timestamp controls (per day, k = min(n_burst_ts, n_control_pool_ts))\n",
        "Control pool = 1 <= n_prints < 4 (excludes n_prints==0; those have no trade prints in enriched table).\n",
        "Random seed: 7 (NumPy default_rng).\n\n",
    ]
    all_burst: list[float] = []
    all_ctrl: list[float] = []
    all_iso: list[float] = []
    rows: list[dict] = []
    for d in DAYS:
        od = _one_day_event_study(d, cnt, te)
        if od is None:
            lines.append(f"day {d}: skip (no burst or no control pool)\n")
            continue
        all_burst.extend(od["brv"].tolist())
        all_ctrl.extend(od["crv"].tolist())
        all_iso.extend(od["srv"].tolist())
        lines.append(
            f"day {d}: k={od['k_matched']} burst_subsampled={od['burst_ts_sampled']}  "
            f"burst n={od['n_burst_prints']} mean={od['mean_burst']:.5g}  "
            f"ctrl n={od['n_ctrl_prints']} mean={od['mean_ctrl']:.5g}  "
            f"Welch_t={od['welch_t_b_minus_c']:.4f}\n"
        )
        lines.append(
            f"  isolated n_prints==1: n={od['n_isolated_prints']} mean={od['mean_isolated']:.5g}\n"
        )
        rows.append(
            {k: v for k, v in od.items() if k not in ("brv", "crv", "srv")}
        )
    a = np.array(all_burst, dtype=float)
    c = np.array(all_ctrl, dtype=float)
    s = np.array(all_iso, dtype=float)
    t_ac = t_welch(a, c) if len(a) > 1 and len(c) > 1 else float("nan")
    t_as = t_welch(a, s) if len(a) > 1 and len(s) > 1 else float("nan")
    lines.append(
        f"\nPooled (all days, all prints at sampled timestamps): "
        f"burst n={len(a)} mean={float(np.mean(a)) if len(a) else float('nan'):.5g}  "
        f"ctrl n={len(c)} mean={float(np.mean(c)) if len(c) else float('nan'):.5g}  "
        f"Welch_t={t_ac:.4f}\n"
    )
    lines.append(
        f"burst vs isolated (all n_prints==1, not resampled): "
        f"isolated n={len(s)} mean={float(np.mean(s)) if len(s) else float('nan'):.5g}  Welch_t={t_as:.4f}\n"
    )
    (OUT / "phase1_burst_event_random_control_summary.txt").write_text("".join(lines), encoding="utf-8")
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "phase1_burst_event_random_control_by_day.csv", index=False)
    print("Wrote burst event study; pooled Welch t burst-ctrl =", t_ac)


def main() -> None:
    p1 = load_p1()
    tr = pd.concat(
        [pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";") for d in DAYS],
        ignore_index=True,
    )
    graph_hubs(tr)
    bf = p1.burst_flags(
        pd.concat(
            [pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS],
            ignore_index=True,
        )
    )
    orchestrator_bursts(bf)
    te = p1.build_trade_enriched()
    tr_all = pd.concat(
        [pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";").assign(day=d) for d in DAYS],
        ignore_index=True,
    )
    burst_event_random_control(te, tr_all)


if __name__ == "__main__":
    main()
