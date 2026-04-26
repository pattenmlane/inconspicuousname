# Round 4 Phase 3 — Sonic gate on tape + inclineGod spreads

- **Script:** `analyze_phase3.py` — R4 inner-join 5200+5300+extract (same as R3 `aligned_panel` + `TH=2`, `K=20`).
- **Key tape result:** `phase3_joint_gate_summary_r4.txt` — all three days show **highly significant** higher mean K-step forward extract mid when **joint tight** vs not (Welch p ≪ 0.001). Day 3 has **P(tight)≈0.46** (more joint-tight time) but the tight-vs-loose **fwd** split remains large.
- **inclineGod:** `phase3_spread_spread_only_r4.csv`, `phase3_spread_mid_correlation_matrix_r4.csv`, figure `phase3_r4_inclineGod_panels.png`.
- **Gate × burst × extract fwd:** `phase3_gate_x_burst_extract_fwd20.csv` — under **joint tight**, non-burst timestamps actually show **higher** mean fwd extract than burst (0.43 vs 0.34); Sonic gate **re-orders** the Phase-2 “burst alone” story.
- **Mark × gate:** `phase3_mark_pair_symbol_gate_markout.csv` — e.g. Mark14→Mark38 hydro: worse mean fwd20 in **tight** than **loose** (tape markout; n modest).
- **Sim:** `trader_v2.py` tests skipping quotes when `market_trades` shows Mark55→Mark01 on extract — **hurts** vs v0/v1 under `worse` (2494 vs 3184).
