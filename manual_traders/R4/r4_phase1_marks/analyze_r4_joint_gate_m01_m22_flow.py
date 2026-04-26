#!/usr/bin/env python3
"""
Round 4 — Mark01→Mark22 **VEV_5300** prints: signed qty vs **fwd_EXTRACT_20**, split by
**joint Sonic gate** (same ``merge_trades_with_tight`` as Phase 3).

Run: python3 manual_traders/R4/r4_phase1_marks/analyze_r4_joint_gate_m01_m22_flow.py
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
    sub = m[
        (m["buyer"] == "Mark 01")
        & (m["seller"] == "Mark 22")
        & (m["symbol"] == "VEV_5300")
    ].copy()
    # aggr sign: buyer lifts ask
    sub["signed"] = np.where(
        sub["side"] == "aggr_buy",
        sub["qty"],
        np.where(sub["side"] == "aggr_sell", -sub["qty"], 0),
    )
    rows = []
    for day in sorted(sub["day"].unique()):
        for tight, lab in [(True, "tight"), (False, "loose")]:
            g = sub[(sub["day"] == day) & (sub["tight"] == tight)]
            x = g["signed"].astype(float).values
            y = g["fwd_EXTRACT_20"].astype(float).values
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 8:
                rows.append(
                    {
                        "day": int(day),
                        "gate": lab,
                        "n": int(ok.sum()),
                        "corr_signed_fwdEXTRACT20": float("nan"),
                    }
                )
                continue
            c = np.corrcoef(x[ok], y[ok])[0, 1]
            rows.append(
                {
                    "day": int(day),
                    "gate": lab,
                    "n": int(ok.sum()),
                    "corr_signed_fwdEXTRACT20": float(c),
                    "mean_fwd_EXTRACT_20": float(np.mean(y[ok])),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase6_m01_m22_vev5300_signed_vs_fwdEXTRACT20_by_gate.csv", index=False)
    lines = [
        "Mark01→Mark22 VEV_5300: corr(signed_qty, fwd_EXTRACT_20) by day and joint gate\n",
        df.to_string(index=False),
        "\n",
    ]
    (OUT / "phase6_m01_m22_flow_summary.txt").write_text("".join(lines), encoding="utf-8")
    print("Wrote phase6_* under", OUT)


if __name__ == "__main__":
    main()
