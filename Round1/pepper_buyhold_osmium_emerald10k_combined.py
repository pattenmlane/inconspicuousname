"""
Round 1 combined submission:
  - INTARIAN_PEPPER_ROOT: buy at best ask each tick until position +80 (buy-and-hold).
  - ASH_COATED_OSMIUM: Emerald-style MM with fair value fixed at 10_000 (take / clear / passive).

Backtest (from ProsperityRepo):
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/pepper_buyhold_osmium_emerald10k_combined.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades all --no-vis
"""
from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
PEPPER_LIMIT = 80
OSMIUM_FV = 10_000.0
OSMIUM_WIDTH = 2
OSMIUM_LIMIT = 80


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

    def emerald_orders_osmium(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
    ) -> List[Order]:
        orders: List[Order] = []

        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2

        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(OSMIUM, int(best_ask), quantity))

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(OSMIUM, int(best_bid), -quantity))

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            OSMIUM,
            sum(o.quantity for o in orders if o.quantity > 0),
            sum(-o.quantity for o in orders if o.quantity < 0),
            fair_value,
            width,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(OSMIUM, int(bbbf + 1), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(OSMIUM, int(baaf - 1), -sell_quantity))

        return orders

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        trader_data = json.dumps({})

        depth_p = state.order_depths.get(PEPPER)
        if depth_p is not None and depth_p.sell_orders:
            pos = int(state.position.get(PEPPER, 0))
            room = PEPPER_LIMIT - pos
            if room > 0:
                best_ask = min(depth_p.sell_orders.keys())
                ask_vol = abs(int(depth_p.sell_orders[best_ask]))
                qty = min(room, ask_vol)
                if qty > 0:
                    result[PEPPER] = [Order(PEPPER, int(best_ask), int(qty))]

        depth_o = state.order_depths.get(OSMIUM)
        if depth_o is not None:
            pos_o = state.position.get(OSMIUM, 0)
            result[OSMIUM] = self.emerald_orders_osmium(
                depth_o,
                OSMIUM_FV,
                OSMIUM_WIDTH,
                pos_o,
                OSMIUM_LIMIT,
            )

        return result, conversions, trader_data
