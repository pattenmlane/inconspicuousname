"""Round 4 placeholder trader — Phase 1 was tape analysis only; strategy wiring in Phase 2."""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, json.dumps({"note": "r4_counterparty_gated phase1 analysis only"}, separators=(",", ":"))
