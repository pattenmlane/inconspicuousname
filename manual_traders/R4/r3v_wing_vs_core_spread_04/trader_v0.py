"""Round 4 baseline — no orders (PnL 0). Counterparty-aware strategies build from Phase 1 outputs."""
from __future__ import annotations

import json
from datamodel import Order, TradingState

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
PRODUCTS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", *[f"VEV_{k}" for k in STRIKES]]


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}
        return {p: [] for p in PRODUCTS}, 0, json.dumps(td)
