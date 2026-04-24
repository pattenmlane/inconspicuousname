"""
Validate osmium_near_fv_bot against fair CSV (tomato validate_bot3 style).

--all-sessions: run both known fair exports + pooled ring χ² on merged events.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path

from osmium_near_fv_bot import (
    NearFvParams,
    fit_params_from_counts,
    near_fv_quote,
    near_fv_quote_empirical,
)
from osmium_near_fv_events import (
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_ABS,
    iter_near_fv_events,
    load_rows_and_fv,
)
from osmium_sessions import all_session_dirs

N_SIMS = 50


def two_sided_binom_z(n_success: int, n_trials: int, p0: float = 0.5) -> tuple[float, float]:
    if n_trials == 0:
        return 0.0, 1.0
    ph = n_success / n_trials
    z = (ph - p0) / math.sqrt(p0 * (1 - p0) / n_trials)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p


def run_session(data_dir: Path, max_abs: float) -> list:
    rows, fv_map, prices_path = load_rows_and_fv(data_dir)
    n_rows = len(rows)
    ev = list(iter_near_fv_events(rows, fv_map, max_abs=max_abs, exclude_mm_offsets=False))
    ts_any = {e.timestamp for e in ev}
    n_ts = len(ts_any)
    n_b = sum(1 for e in ev if e.side == "bid")
    n_a = sum(1 for e in ev if e.side == "ask")
    n_ev = len(ev)

    params = fit_params_from_counts(
        n_timesteps=n_rows,
        n_ts_with_event=n_ts,
        n_bid_events=n_b,
        n_ask_events=n_a,
    )

    print("=" * 70)
    print("  OSMIUM NEAR-FV VALIDATION")
    print(f"  data_dir={data_dir.resolve()}  {prices_path.name}")
    print("=" * 70)
    print(f"\n  Fitted params: p_tick={params.p_tick:.4f}  p_bid={params.p_bid:.3f}")
    print(f"  Actual: timestamps with ≥1 event {n_ts}/{n_rows} ({100*n_ts/n_rows:.2f}%)")
    print(f"  Actual: events={n_ev}  bid={n_b} ask={n_a}")

    z_s, p_s = two_sided_binom_z(n_b, n_ev, 0.5)
    print(f"\n  Side split vs 50/50: z={z_s:.2f}  two-sided p≈{p_s:.2f}")

    ring = {-3, -2, 1, 2}
    sub = [e for e in ev if e.delta in ring]
    if len(sub) >= 8:
        ct = Counter(e.delta for e in sub)
        exp = len(sub) / 4
        chi = sum((ct.get(d, 0) - exp) ** 2 / exp for d in sorted(ring))
        print(f"\n  Ring delta uniform {{-3,-2,+1,+2}}: chi2={chi:.3f} df=3 (n={len(sub)})")

    a_cross = sum(1 for e in ev if e.crossing)
    print(f"\n  Crossing rate actual: {a_cross}/{n_ev} ({100*a_cross/max(n_ev,1):.1f}%)")

    random.seed(42)
    sim_cross: list[float] = []
    sim_counts: list[int] = []
    for _ in range(N_SIMS):
        c = cx = 0
        for row in rows:
            fv = fv_map[int(row["timestamp"])]
            r = near_fv_quote(fv, params=params)
            if r is None:
                continue
            side, price, _vol = r
            c += 1
            if (side == "bid" and price > fv) or (side == "ask" and price < fv):
                cx += 1
        sim_counts.append(c)
        sim_cross.append(cx / c if c else 0.0)

    print(f"  Sim (indep ring) mean events/run: {sum(sim_counts)/len(sim_counts):.1f} (actual {n_ev})")
    print(f"  Sim (indep ring) mean crossing %: {100*sum(sim_cross)/len(sim_cross):.1f}%")

    joint = Counter((e.side, e.delta) for e in ev)
    sim_cross_e: list[float] = []
    for _ in range(N_SIMS):
        cx = c = 0
        for row in rows:
            fv = fv_map[int(row["timestamp"])]
            r = near_fv_quote_empirical(fv, joint, p_tick=params.p_tick)
            if r is None:
                continue
            side, price, _ = r
            c += 1
            if (side == "bid" and price > fv) or (side == "ask" and price < fv):
                cx += 1
        sim_cross_e.append(cx / c if c else 0.0)
    print(
        f"  Sim (empirical joint side,delta) mean crossing %: "
        f"{100*sum(sim_cross_e)/len(sim_cross_e):.1f}%"
    )

    cross_v = [e.vol for e in ev if e.crossing]
    pass_v = [e.vol for e in ev if not e.crossing]
    if cross_v:
        print(f"\n  Crossing vol: mean={sum(cross_v)/len(cross_v):.2f} [{min(cross_v)},{max(cross_v)}]")
    if pass_v:
        print(f"  Passive vol:  mean={sum(pass_v)/len(pass_v):.2f} [{min(pass_v)},{max(pass_v)}]")
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--max-abs", type=float, default=DEFAULT_MAX_ABS)
    ap.add_argument("--all-sessions", action="store_true")
    args = ap.parse_args()
    if args.all_sessions:
        dirs = all_session_dirs()
    elif args.data_dir is not None:
        dirs = (args.data_dir,)
    else:
        dirs = (DEFAULT_DATA_DIR,)

    pooled_ev: list = []
    for d in dirs:
        pooled_ev.extend(run_session(d, args.max_abs))

    if len(dirs) > 1:
        ring = {-3, -2, 1, 2}
        sub = [e for e in pooled_ev if e.delta in ring]
        ct = Counter(e.delta for e in sub)
        exp = len(sub) / 4
        chi = sum((ct.get(d, 0) - exp) ** 2 / exp for d in sorted(ring))
        n_b = sum(1 for e in pooled_ev if e.side == "bid")
        n_ev = len(pooled_ev)
        z_s, p_s = two_sided_binom_z(n_b, n_ev, 0.5)
        print("=" * 70)
        print("  POOLED — merged events from all sessions")
        print("=" * 70)
        print(f"  Total events: {n_ev}  bid share: {100*n_b/n_ev:.1f}%  z vs 50/50: {z_s:.2f} p≈{p_s:.2f}")
        print(f"  Ring χ² uniform {{-3,-2,+1,+2}}: chi2={chi:.3f} df=3 n={len(sub)}")


if __name__ == "__main__":
    main()
