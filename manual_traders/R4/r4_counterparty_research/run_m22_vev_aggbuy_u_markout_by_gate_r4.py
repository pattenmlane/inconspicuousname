#!/usr/bin/env python3
"""
Tape: aggressive buys of VEVs from passive Mark 22 (seller), forward extract markout by Sonic gate.

Uses r4_phase3_trades_with_gate.csv (Phase 3 merge of gate onto Phase-1 enriched trades).
Aggressive buy: aggressor == 'buy'. Passive seller on tape: seller == 'Mark 22'.

Outputs mean/median/n of mark_20_u by tight vs loose and by day.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ENR = Path(__file__).resolve().parent / "outputs" / "r4_phase3_trades_with_gate.csv"
OUT = Path(__file__).resolve().parent / "outputs"


def summarize(g: pd.Series, name: str) -> dict:
    h = g.dropna()
    if len(h) == 0:
        return {"subset": name, "n": 0, "mean": None, "median": None}
    return {
        "subset": name,
        "n": int(len(h)),
        "mean": float(h.mean()),
        "median": float(h.median()),
        "frac_pos": float((h > 0).mean()),
    }


def main() -> None:
    if not ENR.is_file():
        raise SystemExit(f"missing {ENR}; run run_phase3_sonic_gate_r4.py first")
    df = pd.read_csv(ENR)
    v = df[
        (df["symbol"].astype(str).str.startswith("VEV_"))
        & (df["seller"].astype(str) == "Mark 22")
        & (df["aggressor"].astype(str) == "buy")
    ].copy()
    v["mark_20_u"] = pd.to_numeric(v["mark_20_u"], errors="coerce")

    pooled = [
        summarize(v.loc[v["tight"] == True, "mark_20_u"], "tight_gate"),
        summarize(v.loc[v["tight"] == False, "mark_20_u"], "loose_gate"),
    ]
    by_day = []
    for d, g in v.groupby("day"):
        for tight, lab in [(True, "tight"), (False, "loose")]:
            sub = g.loc[g["tight"] == tight, "mark_20_u"]
            if len(sub.dropna()) >= 5:
                by_day.append(
                    {
                        "day": int(d),
                        "gate": lab,
                        **{k: v for k, v in summarize(sub, "").items() if k != "subset"},
                    }
                )

    v2 = df[(df["symbol"].astype(str).str.startswith("VEV_")) & (df["seller"].astype(str) == "Mark 22")].copy()
    v2["mark_20_u"] = pd.to_numeric(v2["mark_20_u"], errors="coerce")
    pooled2 = [
        summarize(v2.loc[v2["tight"] == True, "mark_20_u"], "tight_gate"),
        summarize(v2.loc[v2["tight"] == False, "mark_20_u"], "loose_gate"),
    ]
    diff2 = None
    p2t = pooled2[0]
    p2l = pooled2[1]
    if p2t.get("mean") is not None and p2l.get("mean") is not None:
        diff2 = float(p2t["mean"] - p2l["mean"])

    out = {
        "description": "Aggressive VEV buys (aggressor=buy) with seller Mark 22; mark_20_u = forward extract mid change K=20 rows",
        "n_total_rows": int(len(v)),
        "pooled_by_gate": pooled,
        "by_day_min_n5": by_day,
        "diff_tight_minus_loose_mean": None,
        "all_vev_seller_m22": {
            "n_total_rows": int(len(v2)),
            "pooled_by_gate": pooled2,
            "diff_tight_minus_loose_mean": diff2,
        },
    }
    pt = next((x for x in pooled if x["subset"] == "tight_gate"), None)
    pl = next((x for x in pooled if x["subset"] == "loose_gate"), None)
    if pt and pl and pt.get("mean") is not None and pl.get("mean") is not None:
        out["diff_tight_minus_loose_mean"] = float(pt["mean"] - pl["mean"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r4_m22_vev_aggbuy_u_markout_by_gate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "Aggressive buy VEV (aggressor=buy), seller Mark 22 — U markout K=20 by Sonic gate",
        f"Total prints: {len(v)}",
    ]
    for row in pooled:
        lines.append(
            f"  {row['subset']}: n={row['n']} mean={row['mean']!s} median={row['median']!s} frac_pos={row.get('frac_pos')!s}"
        )
    if out["diff_tight_minus_loose_mean"] is not None:
        lines.append(f"  mean_tight - mean_loose: {out['diff_tight_minus_loose_mean']:.4f}")
    lines.append("")
    lines.append("All VEV prints, seller Mark 22 (any aggressor) — U markout K=20 by gate")
    lines.append(f"Total prints: {len(v2)}")
    for row in pooled2:
        lines.append(
            f"  {row['subset']}: n={row['n']} mean={row['mean']!s} median={row['median']!s} frac_pos={row.get('frac_pos')!s}"
        )
    if diff2 is not None:
        lines.append(f"  mean_tight - mean_loose: {diff2:.4f}")
    (OUT / "r4_m22_vev_aggbuy_u_markout_by_gate.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "r4_m22_vev_aggbuy_u_markout_by_gate.json")


if __name__ == "__main__":
    main()
