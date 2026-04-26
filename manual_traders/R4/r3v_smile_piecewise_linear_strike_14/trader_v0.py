"""
Round 4 scaffold — no trading logic yet; establishes backtest wiring for this folder.

Phase 1 analysis lives in analysis_outputs/ from run_r4_phase1_counterparty.py.
"""
from __future__ import annotations

import json

from datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, json.dumps({}, separators=(",", ":"))
