"""
Round 4 Phase 2 prototype — uses live market_trades (buyer/seller) from TradingState.

Tape evidence (Phase 1/2): aggressive sells by Mark 22 on VEV_5300 associate with negative
short-horizon forward mid (mean fwd_5 ~ -0.15 to -0.19 in tight book cells).

Rule: when this tick includes a market trade on VEV_5300 with seller Mark 22, place a small
aggressive sell at best bid (take liquidity) to lean with the edge. Caps per tick and
respects position limits. No other products (minimal harness test).
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

VEV5300 = "VEV_5300"
LIMIT = 300
QTY = 12


def _book(depth: OrderDepth) -> tuple[int | None, int | None]:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None, None
    return max(buys.keys()), min(sells.keys())


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
        store = _parse_td(getattr(state, "traderData", None))
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}
        mt: dict[str, list[Trade]] = getattr(state, "market_trades", None) or {}

        trigger = False
        for tr in mt.get(VEV5300, []) or []:
            if getattr(tr, "seller", None) == "Mark 22" and int(getattr(tr, "quantity", 0) or 0) > 0:
                trigger = True
                break

        orders: dict[str, list[Order]] = {}
        if trigger and VEV5300 in depths:
            bb, ba = _book(depths[VEV5300])
            if bb is not None and ba is not None:
                p0 = int(pos.get(VEV5300, 0))
                # short: sell at bid (negative quantity)
                qs = min(QTY, max(0, p0 + LIMIT))
                if qs > 0:
                    orders[VEV5300] = [Order(VEV5300, int(bb), -qs)]

        store["last_trigger"] = bool(trigger)
        return orders, 0, json.dumps(store, separators=(",", ":"))
