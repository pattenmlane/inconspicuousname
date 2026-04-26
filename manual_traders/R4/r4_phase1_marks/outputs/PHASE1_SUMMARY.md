# Round 4 Phase 1 — summary (automated)

## Horizon
- **K ∈ {5,20,100}** = forward **price snapshot** steps (unique timestamps per day, +100 per step).

## Participant predictiveness (high level)
- See `participant_markout_by_side_symbol_K.csv`.
- Flag cells with **|t_vs_pool| > 2** and **n≥50** per day-pool for manual review.

## Burst vs extract
- See `burst_vs_extract_fwd20.csv`.

## Baseline residuals
- `baseline_fwd20_residuals.csv` + `top_abs_residual_pairs.csv`.

## Graph
- `directed_pair_counts_notional.csv`, `two_step_chain_counts.txt`.

## Trade prints enriched: **4281** rows (matched to price grid).
