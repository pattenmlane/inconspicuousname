"""
Round 4 — post Phase 3 (Sonic gate analysis complete). No counterparty-conditioned execution in sim yet.
"""
from __future__ import annotations

from datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, state.traderData or ""
