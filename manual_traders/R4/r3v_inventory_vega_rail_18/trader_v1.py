"""
Round 4 — Phase 2 placeholder trader (v1).

Tape analysis for counterparty-conditioned edges is in
run_r4_phase2_analysis.py (no live counterparty hook in state yet).
Orders disabled to avoid untested execution against Phase 2 stats.
"""
from __future__ import annotations

from prosperity4bt.datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, getattr(state, "traderData", "") or ""
