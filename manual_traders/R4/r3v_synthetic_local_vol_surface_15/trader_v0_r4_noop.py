"""Round 4 placeholder trader (no orders) — baseline backtest wiring only."""
from __future__ import annotations

try:
    from datamodel import TradingState
except ImportError:
    from prosperity4bt.datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, getattr(state, "traderData", "") or ""
