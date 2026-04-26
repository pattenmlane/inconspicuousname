# Round 4 Phase 2 — summary

- **Script:** `analyze_phase2.py` (reuses Phase 1 enriched trades).
- **Burst proximity:** ±500 clock units from any `burst_ge4` timestamp; Mark 01→Mark 22 multi-VEV burst flag in `phase2_burst_metadata_by_timestamp.csv`.
- **Regime (tape):** `phase2_regime_tight5300_x_burst_extract_fwd20.csv` uses **VEV_5300 book spread** at each print’s bar (not the printed symbol’s spread).
- **LODO:** `phase2_leave_one_day_burst_extract_welch.csv` — burst vs non-burst extract fwd20 sign **flips negative on day 1** holdout.
- **Microprice:** `phase2_spread_vs_fwd_vol_vev5300_corr.txt` — weak positive corr spread vs next-5 |Δmid|.
- **Sim:** `trader_v1.py` adds extract max spread filter; grid 4/6/8/12 on `worse` → only **≤4** changes behavior (no quotes); **6+** matches `trader_v0`.
