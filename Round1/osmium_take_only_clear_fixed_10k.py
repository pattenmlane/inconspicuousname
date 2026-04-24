"""
ASH_COATED_OSMIUM — **take liquidity only** vs fixed FV = 10_000, plus Emerald-style clears.

Same aggressive rules as Emerald MM:
  - If best_ask < fv → buy at best_ask (lift ask), capped by limit and book.
  - If best_bid > fv → sell at best_bid (hit bid), capped by limit and book.

Then ``clear_position_order``: if still long, sell at ceil(fv) when that bid exists;
if still short, buy at floor(fv) when that ask exists.

**No** passive market-making (no quotes at bbbf+1 / baaf-1).

Backtest:
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/osmium_take_only_clear_fixed_10k.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades all --no-vis
"""
from __future__ import annotations

import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

SYMBOL = "ASH_COATED_OSMIUM"
FAIR_VALUE = 10_000.0
WIDTH = 2
POSITION_LIMIT = 80


class Trader:
    def clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int,
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
    ) -> List[Order]:
        orders: List[Order] = []

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(SYMBOL, int(best_ask), quantity))

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(SYMBOL, int(best_bid), -quantity))

        self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            SYMBOL,
            sum(o.quantity for o in orders if o.quantity > 0),
            sum(-o.quantity for o in orders if o.quantity < 0),
            fair_value,
            width,
        )

        return orders

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        if SYMBOL not in state.order_depths:
            return result, conversions, ""

        position = state.position.get(SYMBOL, 0)
        orders = self.take_orders(
            state.order_depths[SYMBOL],
            FAIR_VALUE,
            WIDTH,
            position,
            POSITION_LIMIT,
        )
        result[SYMBOL] = orders

        return result, conversions, ""
