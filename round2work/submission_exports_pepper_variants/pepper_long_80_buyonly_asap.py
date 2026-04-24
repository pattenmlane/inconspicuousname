"""Copy of ``submissions/pepper_long_80_buyonly_asap.py`` for upload bundles."""

from __future__ import annotations

import json

from datamodel import Order, OrderDepth, TradingState

SYMBOL = "INTARIAN_PEPPER_ROOT"
POSITION_LIMIT = 80


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        trader_data = json.dumps({})

        depth = state.order_depths.get(SYMBOL)
        if depth is None or not depth.sell_orders:
            return result, conversions, trader_data

        pos = int(state.position.get(SYMBOL, 0))
        room = POSITION_LIMIT - pos
        if room <= 0:
            return result, conversions, trader_data

        best_ask = min(depth.sell_orders.keys())
        ask_vol = abs(int(depth.sell_orders[best_ask]))
        qty = min(room, ask_vol)
        if qty <= 0:
            return result, conversions, trader_data

        result[SYMBOL] = [Order(SYMBOL, int(best_ask), int(qty))]
        return result, conversions, trader_data
