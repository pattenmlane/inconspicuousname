"""
Round 4 — placeholder after Phase 2 analysis (no counterparty-conditioned execution yet).

Phase 2 artifacts: `analysis_outputs/phase2/`. Next `trader_v2+` should encode burst-B / adverse
selection rules and backtest with `--match-trades worse` + `all`.
"""
from __future__ import annotations

from datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, state.traderData or ""
