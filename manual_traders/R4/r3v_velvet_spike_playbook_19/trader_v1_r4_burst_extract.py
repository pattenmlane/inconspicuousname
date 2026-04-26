"""
Round 4 — burst-conditioned extract impulse (Phase 2 named-bot exploitation).

Tape: at each tick, `TradingState.market_trades` lists historical prints at this
timestamp (backtester patch). Detect Mark 01 -> Mark 22 multi-VEV basket burst:
>= BURST_MIN_VEV prints on VEV_* with that counterparty pair at the same timestamp.

Phase 2 tape study: mean forward extract mid (+20 price rows) ~0.30 at those prints
(see analysis_outputs/r4_phase2_burst_mark01_22_summary.json). Strategy: small
aggressive buy on VELVETFRUIT_EXTRACT at best ask when burst fires, capped by
position limit.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState, Trade

EXTRACT = "VELVETFRUIT_EXTRACT"
LIMIT = 200
BURST_MIN_VEV = 3
LOT = 6


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def _burst_m01_m22_vev_count(state: TradingState) -> int:
    mt = getattr(state, "market_trades", None) or {}
    n = 0
    for sym, lst in mt.items():
        if not str(sym).startswith("VEV_"):
            continue
        for t in lst or []:
            if getattr(t, "buyer", None) == "Mark 01" and getattr(t, "seller", None) == "Mark 22":
                n += 1
    return n


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths = getattr(state, "order_depths", {}) or {}
        if EXTRACT not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        d = depths[EXTRACT]
        buys = getattr(d, "buy_orders", {}) or {}
        sells = getattr(d, "sell_orders", {}) or {}
        if not buys or not sells:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        burst_n = _burst_m01_m22_vev_count(state)
        store["burst_m01_m22_vev"] = int(burst_n)

        pos = int(getattr(state, "position", {}).get(EXTRACT, 0) or 0)
        out: dict[str, list[Order]] = {}

        if burst_n >= BURST_MIN_VEV:
            ask = min(sells.keys())
            cap = LIMIT - pos
            q = min(LOT, cap, abs(int(sells.get(ask, 0))))
            if q > 0:
                out[EXTRACT] = [Order(EXTRACT, int(ask), int(q))]

        return out, 0, json.dumps(store, separators=(",", ":"))
