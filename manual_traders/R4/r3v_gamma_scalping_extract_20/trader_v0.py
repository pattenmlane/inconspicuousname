"""
Round 4 placeholder trader (counterparty-aware tape available in backtester).

Phase 1 deliverable is **offline tape analysis** (see `analysis.json` and
`analysis_outputs/r4_phase1_*.csv`). This stub posts **no orders** so
`results.json` iteration 0 has an auditable baseline before we wire
counterparty-conditioned rules in v1+.
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV = [f"VEV_{k}" for k in STRIKES]
PRODUCTS = [HYDRO, EXTRACT] + VEV


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        conversions = 0
        store: dict[str, Any] = {}
        obs = getattr(state.observations, "plainValueObservations", None) or {}
        if "__BT_TAPE_DAY__" in obs:
            store["tape_day"] = int(obs["__BT_TAPE_DAY__"])
        return result, conversions, json.dumps(store)
