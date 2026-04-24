"""
INTARIAN_PEPPER_ROOT — Emerald MM with **wall mid** fair value (same idea as
``osmium_mm_emeraldstyle_wallmid.py``):

  bid_wall = min(bid prices), ask_wall = max(ask prices) in the book,
  wall_mid = (bid_wall + ask_wall) / 2.

If the book is one-sided, fall back to **micro mid** (best bid + best ask) / 2, then to
**last_fv** persisted in ``traderData`` JSON (no hardcoded level like 10_000).

Takes / clear / passive ladder match ``pepper_mm_emeraldstyle_drift.py`` (width 2, limit 80).

Backtest:

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/pepper_mm_emeraldstyle_wallmid.py" 1 \\
  --data "$PWD/Prosperity4Data" --match-trades worse --no-vis
"""
from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

SYMBOL = "INTARIAN_PEPPER_ROOT"
WIDTH = 2
POSITION_LIMIT = 80


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (min(depth.buy_orders.keys()) + max(depth.sell_orders.keys())) / 2.0


def _fair(depth: OrderDepth) -> float | None:
    w = _wall_mid(depth)
    if w is not None:
        return float(w)
    return _micro_mid(depth)


class Trader:
    def emerald_orders(
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
                    orders.append(Order(SYMBOL, int(best_ask), quantity))

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(SYMBOL, int(best_bid), -quantity))

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            SYMBOL,
            sum([o.quantity for o in orders if o.quantity > 0]),
            sum([-o.quantity for o in orders if o.quantity < 0]),
            fair_value,
            width,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(SYMBOL, int(bbbf + 1), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(SYMBOL, int(baaf - 1), -sell_quantity))

        return orders

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

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        last = store.get("last_fv")
        last_f = float(last) if isinstance(last, (int, float)) else None

        if SYMBOL not in state.order_depths:
            return result, 0, json.dumps(store)

        depth = state.order_depths[SYMBOL]
        fv = _fair(depth)
        if fv is None:
            if last_f is None:
                return result, 0, json.dumps(store)
            fv = last_f
        else:
            store["last_fv"] = float(fv)

        position = int(state.position.get(SYMBOL, 0))
        orders = self.emerald_orders(
            depth,
            float(fv),
            WIDTH,
            position,
            POSITION_LIMIT,
        )
        result[SYMBOL] = orders
        return result, 0, json.dumps(store)
