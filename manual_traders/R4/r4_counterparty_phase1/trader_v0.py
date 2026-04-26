"""
Round 4 counterparty strategy folder — iteration 0 (baseline).

Phase 1 deliverable is **tape evidence** (see outputs/ and analysis.json). This trader
intentionally places **no orders** so backtests establish a zero-PnL baseline before
Phase-2+ execution rules are wired to Mark-conditioned signals.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import TradingState
except ImportError:
    from prosperity4bt.datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        td: dict[str, Any] = {"note": "r4_counterparty_phase1 baseline no-op"}
        return {}, 0, json.dumps(td, separators=(",", ":"))
