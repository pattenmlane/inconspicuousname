"""
Pepper-only baseline: hold **+80** (target = limit), **no market making**.

Each tick, at most one aggressive order: lift best ask to buy toward +80, or hit
best bid to sell down if above 80. Same inventory logic as
``submissions/test_long80_pepper_osmium_273774style.py`` pepper leg, without osmium.

Compare against ``pepper_70_10_slope_safeguard_standalone.py`` (70/10 drift MM + slope).
"""

from __future__ import annotations

from typing import List

from datamodel import Order, OrderDepth, TradingState

PEPPER = "INTARIAN_PEPPER_ROOT"

PEPPER_TARGET_LONG = 80
PEPPER_POSITION_LIMIT = 80


class Trader:
    def _pepper_long_only_orders(self, depth: OrderDepth, position: int) -> List[Order]:
        if not depth.buy_orders or not depth.sell_orders:
            return []
        lim = max(1, PEPPER_POSITION_LIMIT)
        tgt = max(0, min(PEPPER_TARGET_LONG, lim))
        need = tgt - position
        if need > 0:
            best_ask = min(depth.sell_orders.keys())
            ask_vol = abs(int(depth.sell_orders[best_ask]))
            q = min(need, lim - position, ask_vol)
            if q > 0:
                return [Order(PEPPER, int(best_ask), int(q))]
        if need < 0:
            best_bid = max(depth.buy_orders.keys())
            bid_vol = int(depth.buy_orders[best_bid])
            q = min(-need, lim + position, bid_vol)
            if q > 0:
                return [Order(PEPPER, int(best_bid), -int(q))]
        return []

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        if PEPPER in state.order_depths:
            pos_pe = int(state.position.get(PEPPER, 0))
            result[PEPPER] = self._pepper_long_only_orders(state.order_depths[PEPPER], pos_pe)
        return result, conversions, "{}"
