#!/usr/bin/env python3
"""
Round 4 — Mark **01 → 22** basket prints vs **joint Sonic gate** (tape).

Outputs:
  - ``outputs/phase4_01_22_gate_counts.txt`` — counts tight vs loose per symbol.
  - ``outputs/phase4_01_22_fwd20_joint_tight.csv`` — mean fwd_same_20 when joint_tight (n>=5).

Observation on R4 days 1–3: **all** Mark01→Mark22 prints on VEV_5200/5300/5400 occur at
timestamps where the **inner-join** gate is already tight (loose count = 0).

Run: python3 manual_traders/R4/r4_phase1_marks/analyze_r4_01_22_under_joint_gate.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def main() -> None:
    spec = importlib.util.spec_from_file_location("ap3", HERE / "analyze_phase3.py")
    ap3 = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(ap3)
    p1 = ap3.load_p1()
    m = ap3.merge_trades_with_tight(p1)
    sub = m[(m["buyer"] == "Mark 01") & (m["seller"] == "Mark 22")].copy()

    lines = ["Mark 01 → Mark 22 prints vs joint gate (inner-join tight flag)\n"]
    rows = []
    for sym in ["VEV_5200", "VEV_5300", "VEV_5400"]:
        g = sub[sub["symbol"] == sym]
        nt = int((g["tight"] == True).sum())
        nl = int((g["tight"] == False).sum())
        lines.append(f"{sym}: tight={nt} loose={nl} total={len(g)}\n")
        gt = g[g["tight"] == True]
        x = gt["fwd_same_20"].astype(float).dropna().values
        if len(x) >= 5:
            rows.append(
                {
                    "symbol": sym,
                    "n_joint_tight": len(x),
                    "mean_fwd20": float(np.mean(x)),
                    "frac_pos": float(np.mean(x > 0)),
                }
            )
    (OUT / "phase4_01_22_gate_counts.txt").write_text("".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "phase4_01_22_fwd20_joint_tight.csv", index=False)
    print("Wrote phase4_01_22_gate_counts.txt and phase4_01_22_fwd20_joint_tight.csv")


if __name__ == "__main__":
    main()
