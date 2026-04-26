"""
Round 4 Phase 1 placeholder trader (no live counterparty hook yet).

The backtester does not expose tape `Trade.buyer`/`seller` inside `TradingState` during
`Trader.run`, so counterparty-conditioned execution belongs in a follow-up that either
precomputes signals from `Prosperity4Data/ROUND_4/trades_*.csv` (see
`r4_phase1_counterparty.py` outputs) or extends the runner.

This file exists so `python3 -m prosperity4bt ... 4` can be invoked for a zero-PnL baseline
while Phase 2 implements Mark-conditioned logic.

Products: respect limits if extended later; currently no orders.
"""

from __future__ import annotations

import json
from datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, json.dumps({"note": "R4_phase1_analysis_only_see_outputs_folder"})
