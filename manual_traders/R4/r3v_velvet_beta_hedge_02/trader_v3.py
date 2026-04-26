"""
Round 4 follow-up — passive extract MM when Sonic joint gate is on.

Tape: Phase 3 panel shows extract mid forward K=20 higher when Sonic joint tight vs loose;
Mark 55 aggr-buy split is mixed (see r4_mark55_extract_fwd_by_gate.json). This trader only
provides liquidity on VELVETFRUIT_EXTRACT
(one tick inside) when VEV_5200 and VEV_5300 BBO spreads are both <= 2. No counterparty
filter (avoid adverse taker rules that failed in v1/v2).

Position limit 200 per round4description.txt.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
TH = 2
LIMIT_U = 200
SIZE = 18


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
        _ = _parse_td(getattr(state, "traderData", None))
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}

        s52 = _bbo_spread(depths[V5200]) if V5200 in depths else None
        s53 = _bbo_spread(depths[V5300]) if V5300 in depths else None
        joint = s52 is not None and s53 is not None and s52 <= TH and s53 <= TH

        orders: dict[str, list[Order]] = {}
        if joint and U in depths:
            bb, ba = _book(depths[U])
            if bb is not None and ba is not None and ba > bb + 1:
                bid_p = int(bb) + 1
                ask_p = int(ba) - 1
                if bid_p < ask_p:
                    p0 = int(pos.get(U, 0))
                    qb = min(SIZE, max(0, LIMIT_U - p0))
                    qs = min(SIZE, max(0, LIMIT_U + p0))
                    lo: list[Order] = []
                    if qb > 0:
                        lo.append(Order(U, bid_p, qb))
                    if qs > 0:
                        lo.append(Order(U, ask_p, -qs))
                    if lo:
                        orders[U] = lo

        return orders, 0, json.dumps({"joint": joint}, separators=(",", ":"))
