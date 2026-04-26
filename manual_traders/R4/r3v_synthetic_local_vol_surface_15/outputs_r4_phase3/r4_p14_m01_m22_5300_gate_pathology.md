# Mark 01 → Mark 22 on VEV_5300 (Round 4 days 1–3)

- **Phase-1** `r4_p1_trades_enriched.csv`: 132 rows, **all** have `aggressor_bucket=aggr_sell` (price ≤ bid1).
- Merged with `r4_p3_joint_gate_panel_by_timestamp`: **132/132** have `tight==True` (joint Sonic gate at same timestamp). **0** prints when gate is **loose** for this *pair* on 5300.
- **Implication for Phase-3** “interact with gate” thesis: for this *(buyer, seller, product)*, **Welch tight vs loose is undefined**; any counterfactual “only informative when tight” is **tautological** in-sample (if it only ever fires tight).

Forward mid on the trade **symbol** (from enriched CSV, mean):

| K | mean fwd |
|---|----------|
| 5 | -0.148 |
| 20| -0.121 |
| 100| +0.080 |

- Short horizons align with a **lean short** / fade; K=100 flips (longer-horizon noise).

- Live backtest: `trader_v30_r4_m01_m22_aggr_sell_5300_gate.py` (aggr_sell + gate; short 5300 at bb-1).
