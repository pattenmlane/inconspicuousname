#!/usr/bin/env python3
"""
Union of extract aggressive-buy print timestamps:
- Mark 22 **seller** (`r4_extract_aggr_buy_m22_print.json`)
- Mark 67 **buyer** (`r4_extract_aggr_buy_m67_print.json`)

Output: precomputed/r4_extract_aggr_buy_m22_m67_union_print.json

Requires the two inputs (run preprocess scripts for M22 and M67 first).

Run from repo root:
  python3 manual_traders/R4/r3v_jump_gap_filter_17/preprocess_r4_extract_aggr_buy_m22_m67_union.py
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "precomputed"
M22 = BASE / "r4_extract_aggr_buy_m22_print.json"
M67 = BASE / "r4_extract_aggr_buy_m67_print.json"
OUT = BASE / "r4_extract_aggr_buy_m22_m67_union_print.json"


def main() -> None:
    a = json.loads(M22.read_text(encoding="utf-8"))
    b = json.loads(M67.read_text(encoding="utf-8"))
    out: dict[str, list[int]] = {}
    for k in sorted(set(a) | set(b), key=lambda x: int(x)):
        sa = set(map(int, a.get(k, [])))
        sb = set(map(int, b.get(k, [])))
        out[str(int(k))] = sorted(sa | sb)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    for kk in out:
        print(f"day {kk}: union={len(out[kk])}")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
