#!/usr/bin/env python3
"""
Build **Phase 1 compliance** tables for the expanded ops checklist:

1. **phase1_participant_print_counts.csv** — every distinct `U` in ROUND_4 tape:
   rows as buyer, as seller, total prints, and **which products** appear (string).

2. **PHASE1_PING_CHECKLIST.md** — maps each `suggested direction.txt` Phase 1 bullet
   to **existing** outputs in ``manual_traders/R4/r4_phase1_marks/outputs/`` and
   one-line pointers to `analyze_phase1.py` sections.

**Horizon note (R4, matches analyze_phase1):** K \u2208 {5,20,100} = **unique
timestamp index** within day (+100 per step on tape), *not* wall-clock seconds.

Run: python3 manual_traders/R4/r4_phase1_marks/analyze_r4_phase1_compliance_table.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
OUT = HERE.parent / "outputs"
REPO = HERE.parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
DAYS = [1, 2, 3]


def main() -> None:
    names_buy: dict[str, int] = defaultdict(int)
    names_sell: dict[str, int] = defaultdict(int)
    name_syms: dict[str, set[str]] = defaultdict(set)

    for d in DAYS:
        tr = pd.read_csv(DATA / f"trades_round_4_day_{d}.csv", sep=";")
        for _, r in tr.iterrows():
            b = str(r.get("buyer", ""))
            s = str(r.get("seller", ""))
            sym = str(r.get("symbol", ""))
            if b and b != "nan":
                names_buy[b] += 1
                name_syms[b].add(sym)
            if s and s != "nan":
                names_sell[s] += 1
                name_syms[s].add(sym)

    all_marks = sorted(set(names_buy) | set(names_sell))
    rows = []
    for u in all_marks:
        nb, ns = names_buy.get(u, 0), names_sell.get(u, 0)
        syms = ",".join(sorted(name_syms[u]))
        rows.append(
            {
                "participant_U": u,
                "n_prints_as_buyer": nb,
                "n_prints_as_seller": ns,
                "n_prints_total": nb + ns,
                "products_touched": syms,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase1_participant_print_counts.csv", index=False)

    check = f"""# Round 4 Phase 1 — operations checklist (repo mapping)

**Tape days present:** {", ".join(str(d) for d in DAYS)} (all files under `Prosperity4Data/ROUND_4/`).  
**Participant fields:** `buyer`, `seller` on trades (see `round4work/round4description.txt`).

**Forward horizon K \u2208 {{5, 20, 100}}:** **price-bar / timestamp steps** (one row per unique
`timestamp` in the day\u2019s price CSV, advancing one step = one snapshot). Documented in
`outputs/PHASE1_SUMMARY.md` and `analyze_phase1.py` header.

---

## Bullet 1 — Participant-level alpha / predictiveness

| Deliverable | Primary outputs | Implementation |
|-------------|-----------------|----------------|
| Per-name U, aggressor side, same-symbol fwd K | `participant_markout_by_side_symbol_K.csv` | `analyze_phase1::participant_tables` — Welch vs spread-matched pool |
| Median, frac pos, n | same CSV | \u2014 |
| Cross-asset fwd **VELVETFRUIT_EXTRACT** / **HYDROGEL_PACK** | columns `mean_fwd_EXTRACT_K`, `mean_fwd_HYDRO_K` in same file | \u2014 |
| Stratify: spread (tight/mid/wide), session tertile | `stratified_cell_means.csv` | `stratified_cell_stats` (symbol \u00d7 spread_bin \u00d7 session \u00d7 side) |
| **Coverage:** distinct participants on tape | `phase1_participant_print_counts.csv` (this run) | full enumeration |

**Conclusion (unchanged):** no |t| \u2265 2 with n\u226540 vs pool for strong cells; R4 days 1\u20113 only.

---

## Bullet 2 — Deviation from bot baseline

| Deliverable | Primary outputs |
|-------------|-----------------|
| Cell mean by (buyer, seller, symbol, spread_bin) + residual | `baseline_fwd20_residuals.csv` |
| Top |residual| pairs | `top_abs_residual_pairs.csv` |

`analyze_phase1::bot_baseline_residuals`

---

## Bullet 3 — Graph / lead-lag

| Deliverable | Primary outputs |
|-------------|-----------------|
| Directed pair counts + notional | `directed_pair_counts_notional.csv` |
| 2-step tape chains | `two_step_chain_counts.txt` |
| Lagged signed flow vs extract fwd | `lagged_signed_flow_extract_corr.csv` |

`analyze_phase1::graph_pairs_notional`, `lagged_flow_extract`

---

## Bullet 4 — Bursts

| Deliverable | Primary outputs |
|-------------|-----------------|
| Burst vs non burst extract fwd20 | `burst_vs_extract_fwd20.csv` |
| Welch ge4 vs lt4 | `burst_extract_welch_ge4_vs_lt4.txt` |

`analyze_phase1::burst_event_study` (uses `burst_flags`: same (day, timestamp) multi-row)

---

## Bullet 5 — Adverse selection proxy

| Deliverable | Primary outputs |
|-------------|-----------------|
| Per-party on **extract** prints, mean fwd20 | `extract_passive_party_fwd20_proxy.csv` |

`analyze_phase1::passive_markout_proxy`

---

## Authoritative `round4_phase1_complete` in `analysis.json`

The structured gate object (bullets 1\u20135, top-5 ranked, no-edge list) is **already** in
`analysis.json` as element containing `"round4_phase1_complete": true`. **Later
iterations** (phase8\u2013phase12) add **hydro duopoly \u00d7 Sonic gate** evidence that **refines**
the hydro bullet but do not replace the core Phase-1 `analyze_phase1` sweep.

---
Generated by `analyze_r4_phase1_compliance_table.py`
"""
    (OUT / "PHASE1_PING_CHECKLIST.md").write_text(check, encoding="utf-8")
    print("Wrote", OUT / "PHASE1_PING_CHECKLIST.md")
    print("Wrote", OUT / "phase1_participant_print_counts.csv", "n_participants=", len(df))


if __name__ == "__main__":
    main()
