"""
Round 4 placeholder trader (iteration 0). Phase 1 delivered tape analysis only;
implement counterparty-conditioned strategy in trader_v1+ after sim design.
"""
from __future__ import annotations

from datamodel import Order, TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, ""
