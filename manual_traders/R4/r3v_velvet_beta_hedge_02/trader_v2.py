"""
Round 4 Phase 3 — Sonic joint gate + counterparty (tape-informed).

When BBO spreads on VEV_5200 and VEV_5300 are both <= 2 AND this tick's market_trades
include Mark 22 selling VEV_5300, place a small aggressive sell at best bid (same intent
as v1 but only in tight-surface regime — Phase 3 shows all Mark22 aggr sells on 5300 occur
under tight gate; n_loose=0 in r4_phase3_counterparty_x_gate_extract_5300.json).
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

VEV5200 = "VEV_5200"
VEV5300 = "VEV_5300"
TH = 2
LIMIT = 300
QTY = 10


def _bbo_spread(depth: OrderDepth) -> int | None:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if ba <= bb:
        return None
    return int(ba - bb)


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

        s52 = _bbo_spread(depths[VEV5200]) if VEV5200 in depths else None
        s53 = _bbo_spread(depths[VEV5300]) if VEV5300 in depths else None
        joint = s52 is not None and s53 is not None and s52 <= TH and s53 <= TH

        m22_sell = False
        for tr in mt.get(VEV5300, []) or []:
            if getattr(tr, "seller", None) == "Mark 22" and int(getattr(tr, "quantity", 0) or 0) > 0:
                m22_sell = True
                break

        orders: dict[str, list[Order]] = {}
        if joint and m22_sell and VEV5300 in depths:
            bb, ba = _book(depths[VEV5300])
            if bb is not None and ba is not None:
                p0 = int(pos.get(VEV5300, 0))
                qs = min(QTY, max(0, p0 + LIMIT))
                if qs > 0:
                    orders[VEV5300] = [Order(VEV5300, int(bb), -qs)]

        store["joint"] = joint
        store["m22"] = m22_sell
        return orders, 0, json.dumps(store, separators=(",", ":"))
