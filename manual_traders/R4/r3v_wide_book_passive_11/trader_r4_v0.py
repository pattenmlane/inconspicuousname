"""Round 4 baseline trader (no orders) — for wiring backtester + results.json only.

Counterparty-aware strategy work is driven by tape analysis in this folder;
see analysis_outputs_r4_phase1/ and analysis.json (round4_phase1_complete).
"""

from __future__ import annotations

import json
from typing import Dict, List

from datamodel import Order, TradingState

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}


class Trader:
    def run(self, state: TradingState):
        return {p: [] for p in LIMITS}, 0, json.dumps({})
