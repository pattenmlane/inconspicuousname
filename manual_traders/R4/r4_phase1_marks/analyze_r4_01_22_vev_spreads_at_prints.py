#!/usr/bin/env python3
"""BBO spreads at Mark01→Mark22 prints (joint-gate book columns)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
HERE = FILE.parent
DATA = FILE.parents[3] / "Prosperity4Data" / "ROUND_4"
DAYS = [1, 2, 3]


def main() -> None:
    spec = importlib.util.spec_from_file_location("ap3", HERE / "analyze_phase3.py")
    ap3 = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(ap3)

    rows = []
    for day in DAYS:
        p = ap3.add_forward_and_tight(ap3.aligned_panel_r4(day))
        tr = pd.read_csv(DATA / f"trades_round_4_day_{day}.csv", sep=";")
        tr["day"] = day
        sub = tr[(tr["buyer"] == "Mark 01") & (tr["seller"] == "Mark 22")]
        mg = sub.merge(
            p[["timestamp", "s5200", "s5300", "tight"]],
            left_on="timestamp",
            right_on="timestamp",
            how="inner",
        )
        for sym in ["VEV_5200", "VEV_5300", "VEV_5400"]:
            g = mg[mg["symbol"] == sym]
            if len(g) == 0:
                continue
            rows.append(
                {
                    "day": day,
                    "symbol": sym,
                    "n": len(g),
                    "mean_s5200": float(g["s5200"].mean()),
                    "mean_s5300": float(g["s5300"].mean()),
                    "mean_s_gate_leg": float(g["s5200"].mean())
                    if sym == "VEV_5200"
                    else float(g["s5300"].mean())
                    if sym == "VEV_5300"
                    else float("nan"),
                }
            )
    out = HERE / "outputs" / "phase5_01_22_bbo_spreads_at_prints.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    lines = ["Mark01→Mark22: mean BBO spreads at print timestamp (from aligned panel)\n"]
    for _, r in pd.DataFrame(rows).iterrows():
        lines.append(
            f"day{r['day']} {r['symbol']}: n={int(r['n'])} mean_s5200={r['mean_s5200']:.3f} mean_s5300={r['mean_s5300']:.3f}\n"
        )
    (HERE / "outputs" / "phase5_01_22_bbo_spreads_at_prints.txt").write_text("".join(lines), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
