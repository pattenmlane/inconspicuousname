"""Round 4 placeholder trader — Phase 1 was tape-only counterparty research.

Next iterations will implement counterparty-conditioned logic from
manual_traders/R4/r4_counterparty_phase1/r4_phase1_markout_summary.json.
"""
from __future__ import annotations
import json
from datamodel import Order, TradingState

LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
}


class Trader:
    def run(self, state: TradingState):
        return {p: [] for p in LIMITS}, 0, json.dumps({})
