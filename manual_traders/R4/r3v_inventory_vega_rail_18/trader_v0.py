"""
Round 4 — baseline submission trader (v0).

No orders; establishes backtest wiring and PnL=0 reference for Round 4.
Phase 1 analysis is tape-only in run_r4_phase1_counterparty_analysis.py.
"""
from __future__ import annotations

from prosperity4bt.datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, getattr(state, "traderData", "") or ""
