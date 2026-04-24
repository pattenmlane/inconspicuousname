#!/usr/bin/env python3
"""
Print z-signal metrics for fixed (Wz, Ws) pairs on **ASH_COATED_OSMIUM** (jmerle mid,
days overlapping Prosperity 3 round1 ink: -2,-1,0).

Optional: ``--backtest`` runs ``Round1/osmium_jmerle_squidstyle_zscore.py`` on round 1
with ``--match-trades worse`` for each pair (slow).

Usage:
  python3 Prosperity4Data/grid_osmium_jmerle_z_windows.py
  python3 Prosperity4Data/grid_osmium_jmerle_z_windows.py --backtest
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from compare_zsignal_ink_p3_vs_osmium_p4 import (  # noqa: E402
    discover_days,
    metrics_for_window,
    mid_series,
)

INK_ROOT = _REPO / "Prosperity3Data" / "round1"
OSM_ROOT = _REPO / "Prosperity4Data" / "ROUND1"
OSM_PRODUCT = "ASH_COATED_OSMIUM"
TRADER = _REPO / "Round1" / "osmium_jmerle_squidstyle_zscore.py"

DEFAULT_WINDOWS = [(104, 88), (112, 80), (98, 54), (72, 68), (86, 44)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backtest", action="store_true", help="Run prosperity4bt worse for each window (slow).")
    p.add_argument("--thresh", type=float, default=1.0)
    args = p.parse_args()

    ink_days = set(discover_days(INK_ROOT))
    osm_days = sorted(d for d in discover_days(OSM_ROOT) if d in ink_days)
    df = mid_series(OSM_ROOT, osm_days, OSM_PRODUCT, "jmerle")

    print(f"{OSM_PRODUCT} jmerle mid — days {osm_days} n={len(df)}  threshold=±{args.thresh}")
    print()
    for wz, ws in DEFAULT_WINDOWS:
        m = metrics_for_window(df, wz, ws, args.thresh)
        if m is None:
            print(f"Wz={wz} Ws={ws}: insufficient data")
            continue
        print(
            f"Wz={wz:3d} Ws={ws:3d}  corr(sig,f1)={m['corr_f1']:+.4f}  MR_score={m['mr_score']:+.4f}  "
            f"sym_f1={m['sym_f1']:.4f}"
        )
    print()

    if not args.backtest:
        return

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = (
        f"{_REPO / 'imc-prosperity-4-backtester'}:{_REPO / 'imc-prosperity-4-backtester' / 'prosperity4bt'}"
        + (":" + env_base["PYTHONPATH"] if env_base.get("PYTHONPATH") else "")
    )
    cmd = [
        sys.executable,
        "-m",
        "prosperity4bt",
        str(TRADER),
        "1",
        "--data",
        str(_REPO / "Prosperity4Data"),
        "--match-trades",
        "worse",
        "--no-out",
        "--no-vis",
        "--no-progress",
    ]
    print("=== backtest worse, round 1 (all days in data) ===")
    for wz, ws in DEFAULT_WINDOWS:
        env = {**env_base, "OSMIUM_JMERLE_WZ": str(wz), "OSMIUM_JMERLE_WS": str(ws)}
        print(f"\n--- OSMIUM_JMERLE_WZ={wz} OSMIUM_JMERLE_WS={ws} ---", flush=True)
        r = subprocess.run(cmd, cwd=str(_REPO), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:] if r.stderr else "(no stderr)")
            continue
        lines = r.stdout.strip().splitlines()
        for line in lines[-8:]:
            print(line)


if __name__ == "__main__":
    main()
