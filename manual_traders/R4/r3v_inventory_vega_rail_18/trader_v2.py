"""
Round 4 — Phase 3 placeholder (v2).

Tape: Phase 3 joint gate + three-way tables in run_r4_phase3_joint_gate_analysis.py.
No orders until strategy encodes gate × counterparty rules with sim PnL.
"""
from __future__ import annotations

from prosperity4bt.datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, getattr(state, "traderData", "") or ""
