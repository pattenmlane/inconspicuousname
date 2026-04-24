#!/usr/bin/env python3
"""
Deep analysis: near-FV / residual participant (tomato Bot 3 style).

Identification: any book level with |price - true_fv| <= MAX_ABS (default 4),
same continuous threshold as tomato (inside inner MM spread in tick space).

Writes: osmium_near_fv_analysis.txt in --data-dir.

--all-sessions: run once per session + write osmium_near_fv_analysis_MULTISESSION.txt.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from osmium_near_fv_events import (
    DEFAULT_DATA_DIR,
    DEFAULT_MAX_ABS,
    iter_near_fv_events,
    load_rows_and_fv,
)
from osmium_sessions import all_session_dirs

HERE_NEAR = Path(__file__).resolve().parent


def frac_bin(x: float, step: float = 0.05) -> float:
    f = x - math.floor(x)
    b = round(f / step) * step
    if b >= 1.0:
        b = 0.0
    return b


def pearson_chi2_independence(
    cat_a: list[str], cat_b: list[str],
) -> tuple[float, int]:
    """2-way chi-square; returns (chi2, df)."""
    n = len(cat_a)
    if n == 0:
        return 0.0, 0
    rows = sorted(set(cat_a))
    cols = sorted(set(cat_b))
    obs = defaultdict(int)
    for a, b in zip(cat_a, cat_b):
        obs[(a, b)] += 1
    r_m = {r: sum(obs.get((r, c), 0) for c in cols) for r in rows}
    c_m = {c: sum(obs.get((r, c), 0) for r in rows) for c in cols}
    chi2 = 0.0
    for r in rows:
        for c in cols:
            o = obs.get((r, c), 0)
            e = r_m[r] * c_m[c] / n
            if e > 0:
                chi2 += (o - e) ** 2 / e
    df = max((len(rows) - 1) * (len(cols) - 1), 0)
    return chi2, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--max-abs", type=float, default=DEFAULT_MAX_ABS)
    ap.add_argument("--exclude-mm", action="store_true", help="Drop levels whose off_int is inner/wall")
    ap.add_argument("--all-sessions", action="store_true")
    args = ap.parse_args()
    if args.all_sessions:
        chunks: list[str] = []
        for d in all_session_dirs():
            cmd = [
                sys.executable,
                str(HERE_NEAR / "analyze_osmium_near_fv.py"),
                "--data-dir",
                str(d),
                "--max-abs",
                str(args.max_abs),
            ]
            if args.exclude_mm:
                cmd.append("--exclude-mm")
            subprocess.run(cmd, check=True, cwd=str(HERE_NEAR))
            p = d / "osmium_near_fv_analysis.txt"
            chunks.append(f"{'=' * 20}  {d.resolve()}  {'=' * 20}\n\n{p.read_text(encoding='utf-8')}")
        combo = HERE_NEAR / "osmium_near_fv_analysis_MULTISESSION.txt"
        combo.write_text("\n\n".join(chunks), encoding="utf-8")
        print(f"Wrote per-session + {combo}")
        return

    rows, fv_map, prices_path = load_rows_and_fv(args.data_dir)
    n_ts = len(rows)

    ev_all = list(iter_near_fv_events(rows, fv_map, max_abs=args.max_abs, exclude_mm_offsets=False))
    ev_res = list(
        iter_near_fv_events(rows, fv_map, max_abs=args.max_abs, exclude_mm_offsets=True)
    )

    def section(title: str, ev: list, label: str) -> list[str]:
        L: list[str] = []
        W = L.append
        W("")
        W("=" * 72)
        W(f"  {title}")
        W("=" * 72)
        n_ev = len(ev)
        W(f"  Slice: {label}")
        W(f"  prices file: {prices_path.name}  timesteps={n_ts}  events={n_ev}")

        ts_with_bid = {e.timestamp for e in ev if e.side == "bid"}
        ts_with_ask = {e.timestamp for e in ev if e.side == "ask"}
        ts_any = ts_with_bid | ts_with_ask
        ts_both = ts_with_bid & ts_with_ask
        W(f"\n  Timestamps with ≥1 near-FV event: {len(ts_any)}/{n_ts} ({100*len(ts_any)/n_ts:.2f}%)")
        W(f"  Timestamps with bid-side event:   {len(ts_with_bid)}")
        W(f"  Timestamps with ask-side event:   {len(ts_with_ask)}")
        W(f"  Timestamps with BOTH sides:       {len(ts_both)} ({100*len(ts_both)/max(n_ts,1):.2f}%)")

        # Events per timestamp
        ect = Counter()
        for e in ev:
            ect[e.timestamp] += 1
        W(f"\n  Events per timestamp:")
        per = Counter(ect.values())
        for k in sorted(per.keys()):
            W(f"    {k} event(s): {per[k]} timestamps")

        # Run lengths in file row order (one row = one timestep snapshot)
        has_ev = [int(r["timestamp"]) in ts_any for r in rows]
        runs_on: list[int] = []
        runs_off: list[int] = []
        if has_ev:
            cur = has_ev[0]
            run = 1
            for j in range(1, len(has_ev)):
                if has_ev[j] == cur:
                    run += 1
                else:
                    (runs_on if cur else runs_off).append(run)
                    cur = has_ev[j]
                    run = 1
            (runs_on if cur else runs_off).append(run)
        if runs_on:
            rc = Counter(runs_on)
            W(f"\n  Presence run lengths (contiguous timesteps in file order):")
            W(f"    ON runs: n={len(runs_on)} mean={sum(runs_on)/len(runs_on):.2f}")
            for ln in sorted(rc)[:12]:
                W(f"      len {ln}: {rc[ln]}")
            pct1 = 100 * rc.get(1, 0) / len(runs_on) if runs_on else 0
            W(f"    fraction of ON-runs with length 1: {pct1:.1f}%")
        if runs_off:
            W(f"    OFF runs: n={len(runs_off)} mean={sum(runs_off)/len(runs_off):.2f}")

        # Side / delta / crossing
        W(f"\n  EVENT-LEVEL (each level is one event)")
        n_b = sum(1 for e in ev if e.side == "bid")
        n_a = sum(1 for e in ev if e.side == "ask")
        W(f"    Bid events: {n_b}  Ask events: {n_a}  ({100*n_b/max(n_ev,1):.1f}% / {100*n_a/max(n_ev,1):.1f}%)")

        d_ct = Counter(e.delta for e in ev)
        W(f"\n  delta = price - round(FV):")
        for d in sorted(d_ct):
            W(f"    {d:+d}: {d_ct[d]} ({100*d_ct[d]/max(n_ev,1):.1f}%)")

        W(f"\n  (SIDE, DELTA) joint counts:")
        jt = Counter((e.side, e.delta) for e in ev)
        for key in sorted(jt):
            W(f"    {key}: {jt[key]}")

        offi = Counter(e.off_int for e in ev)
        W(f"\n  off_int = round(price - FV) (among |cont|<=max):")
        for o in sorted(offi)[:25]:
            W(f"    {o:+d}: {offi[o]}")
        if len(offi) > 25:
            W(f"    ... ({len(offi)} distinct values)")

        cr = sum(1 for e in ev if e.crossing)
        W(f"\n  Crossing (bid price>FV or ask price<FV): {cr}/{n_ev} ({100*cr/max(n_ev,1):.1f}%)")

        cross_v = [e.vol for e in ev if e.crossing]
        pass_v = [e.vol for e in ev if not e.crossing]
        W(f"\n  VOLUME | CROSSING (tomato-style)")
        if cross_v:
            W(
                f"    Crossing: n={len(cross_v)} mean={sum(cross_v)/len(cross_v):.2f} "
                f"min={min(cross_v)} max={max(cross_v)}"
            )
        else:
            W("    Crossing: n=0")
        if pass_v:
            W(
                f"    Passive:  n={len(pass_v)} mean={sum(pass_v)/len(pass_v):.2f} "
                f"min={min(pass_v)} max={max(pass_v)}"
            )
        else:
            W("    Passive: n=0")

        W(f"\n  VOLUME | DELTA")
        by_d: dict[int, list[int]] = defaultdict(list)
        for e in ev:
            by_d[e.delta].append(e.vol)
        for d in sorted(by_d):
            vs = by_d[d]
            W(f"    delta {d:+d}: n={len(vs)} mean_vol={sum(vs)/len(vs):.2f}")

        # Volume | (side, crossing)
        W(f"\n  VOLUME | (SIDE, CROSSING)")
        for side in ("bid", "ask"):
            for cx in (True, False):
                vs = [e.vol for e in ev if e.side == side and e.crossing == cx]
                if vs:
                    W(
                        f"    {side} {'cross' if cx else 'pass'}: n={len(vs)} "
                        f"mean={sum(vs)/len(vs):.2f} [{min(vs)},{max(vs)}]"
                    )

        # Chi^2: tomato {-2,-1,0,+1} (usually wrong for osmium — mass at ±2,±3)
        allowed_tomato = {-2, -1, 0, 1}
        sub_t = [e for e in ev if e.delta in allowed_tomato]
        if sub_t:
            ct = Counter(e.delta for e in sub_t)
            k = 4
            exp = len(sub_t) / k
            chi = sum((ct.get(d, 0) - exp) ** 2 / exp for d in sorted(allowed_tomato))
            W(
                f"\n  Chi-squared vs TOMATO uniform on delta in {{-2,-1,0,+1}}: "
                f"chi2={chi:.3f} df=3 crit(0.05)=7.815"
            )
            W(f"    (using {len(sub_t)}/{n_ev} events with delta in that set)")
        # Osmium "ring": mostly {-3,-2,+1,+2} (avoids at-the-money)
        ring = {-3, -2, 1, 2}
        sub_r = [e for e in ev if e.delta in ring]
        if sub_r:
            ct = Counter(e.delta for e in sub_r)
            exp = len(sub_r) / 4
            chi_r = sum((ct.get(d, 0) - exp) ** 2 / exp for d in sorted(ring))
            W(
                f"  Chi-squared vs uniform on OSMIUM ring {{-3,-2,+1,+2}}: "
                f"chi2={chi_r:.3f} df=3 crit(0.05)=7.815"
            )
            W(f"    (using {len(sub_r)}/{n_ev} events)")

        # Independence delta bucket vs crossing (coarse)
        if n_ev >= 20:
            da = ["neg" if e.delta < 0 else ("zero" if e.delta == 0 else "pos") for e in ev]
            cb = ["cross" if e.crossing else "pass" for e in ev]
            chi_i, df_i = pearson_chi2_independence(da, cb)
            W(f"\n  Pearson chi2 delta sign bucket x crossing: chi2={chi_i:.3f} df={df_i}")

        # FV fractional part vs presence
        pres_by_frac = defaultdict(lambda: {"p": 0, "t": 0})
        for row in rows:
            ts = int(row["timestamp"])
            fv = fv_map[ts]
            fb = frac_bin(fv)
            pres_by_frac[fb]["t"] += 1
            if ts in ts_any:
                pres_by_frac[fb]["p"] += 1
        W(f"\n  Presence rate by FV fractional bin (0.05 steps, first bins):")
        for fb in sorted(pres_by_frac)[:8]:
            d = pres_by_frac[fb]
            if d["t"]:
                W(f"    frac~{fb:.2f}: present {d['p']}/{d['t']} ({100*d['p']/d['t']:.1f}%)")

        return L

    lines: list[str] = []
    W = lines.append
    W("OSMIUM NEAR-FV / RESIDUAL ANALYSIS")
    W(f"data_dir={args.data_dir.resolve()}  max_abs={args.max_abs}")
    lines += section("ALL LEVELS with |price - FV| <= max_abs", ev_all, "includes any cluster inside band")
    lines += section(
        "RESIDUAL (exclude off_int in {-11,-10,-8,8,10,11})",
        ev_res,
        "MM integer offsets removed — stricter participant slice",
    )

    out = args.data_dir.resolve() / "osmium_near_fv_analysis.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
