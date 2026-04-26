"""
Round 4 placeholder trader (iteration 0): no orders — establishes backtest wiring and
`results.json` / `analysis.json` layout under manual_traders/R4/r3v_jump_gap_filter_17/.

Counterparty-conditioned strategy will land in v1+; Phase 1 tape analysis is in
`r4_phase1_counterparty_analysis.py` and outputs/phase1/.
"""
from __future__ import annotations

import json

try:
    from datamodel import Order, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, TradingState

PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]


class Trader:
    def run(self, state: TradingState):
        out: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        return out, 0, json.dumps({"r4": {"note": "v0 no-op"}}, separators=(",", ":"))
