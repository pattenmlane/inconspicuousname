"""
Round 4 placeholder trader (iteration 0).

Phase 1 deliverable for this commit is **tape analysis** (see `r4_phase1_counterparty_analysis.py`
and `analysis_outputs/`). This module exists so the Round 4 folder matches the expected artifact
layout; strategy logic will be added after Phase 1 edges are validated in sim.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import TradingState


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        return {}, 0, json.dumps(td, separators=(",", ":"))
