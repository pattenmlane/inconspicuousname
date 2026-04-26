#!/usr/bin/env python3
"""Per-day stability: extract fwd20 mean tight vs loose (Sonic gate), Round 4 days 1-3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "manual_traders/R4/r4_counterparty_phase1/analyze_r4_phase3_sonic_gate_r4.py"
OUT = Path(__file__).resolve().parent / "r4_phase3_gate_extract_by_day.json"

# Inline: duplicate minimal logic filtering by day — import main pieces
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_r4_phase3_sonic_gate_r4 as g

DAYS = (1, 2, 3)
K = 20


def fwd_for_day(rows_day: dict[int, dict], day: int) -> tuple[list[float], list[float]]:
    """Return (fwd_tight, fwd_loose) for extract K rows ahead."""
    rows = {ts: r for ts, r in rows_day.items()}
    if not rows:
        return [], []
    tss = sorted(rows)
    mids = [rows[t]["m_ext"] for t in tss]
    tight_l, loose_l = [], []
    for ts in tss:
        fk = g.fwd_delta(tss, mids, ts, K)
        if fk is None:
            continue
        if rows[ts]["tight"]:
            tight_l.append(fk)
        else:
            loose_l.append(fk)
    return tight_l, loose_l


def main() -> None:
    by_day: dict[int, dict[int, dict]] = {d: {} for d in DAYS}
    for day in DAYS:
        rows = g.load_aligned(day)
        by_day[day] = rows
    out = {"per_day": [], "K": K}
    for day in DAYS:
        t, lo = fwd_for_day(by_day[day], day)
        mt = sum(t) / len(t) if t else None
        ml = sum(lo) / len(lo) if lo else None
        _, diff = g.welch_t(t, lo) if t and lo else (None, None)
        out["per_day"].append(
            {
                "day": day,
                "n_tight": len(t),
                "n_loose": len(lo),
                "mean_fwd20_tight": round(mt, 6) if mt is not None else None,
                "mean_fwd20_loose": round(ml, 6) if ml is not None else None,
                "mean_diff_tight_minus_loose": round(diff, 6) if diff is not None else None,
            }
        )
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
