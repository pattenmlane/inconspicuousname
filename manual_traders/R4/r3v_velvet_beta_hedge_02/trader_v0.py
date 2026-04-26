"""
Round 4 placeholder — counterparty-aware strategy to be built from Phase 1 outputs.

Respects product scope; returns no orders until wired to signals.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import TradingState
except ImportError:
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
        _ = _parse_td(getattr(state, "traderData", None))
        return {}, 0, json.dumps({}, separators=(",", ":"))
