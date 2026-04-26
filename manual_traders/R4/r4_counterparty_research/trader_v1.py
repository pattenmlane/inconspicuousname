"""Round 4 Phase 3 milestone placeholder (analysis in outputs/r4_phase3_*) trader: no orders (Phase 1 is offline tape analysis only)."""
from __future__ import annotations

import json
from datamodel import TradingState
from typing import Any


class Trader:
    def run(self, state: TradingState):
        return {}, 0, json.dumps({}, separators=(",", ":"))
