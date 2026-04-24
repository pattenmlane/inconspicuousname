"""
ASH_COATED_OSMIUM — Emerald-style MM (same take / clear / passive as fixed-FV osmium),
but **fair value = rolling mean of the last N micro-mids** (default N=30), after appending
this row’s mid. Warm-up: FV is the mean of all mids so far if fewer than N exist.

Override window: ``OSMIUM_ROLL_WINDOW=20`` (any integer ≥ 1).

Otherwise identical to ``osmium_mm_emeraldstyle_fixed_10k.py`` (no JSON intent logging).

Backtest:
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/osmium_mm_emeraldstyle_roll30_fv.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades all --no-vis

Grid (example):
  for w in 50 30 20 15 10 5; do OSMIUM_ROLL_WINDOW=$w python3 -m prosperity4bt ...; done
"""
from __future__ import annotations

import json
import math
import os
from typing import List

from datamodel import Order, OrderDepth, TradingState

SYMBOL = "ASH_COATED_OSMIUM"
WIDTH = 2
POSITION_LIMIT = 80
HISTORY_CAP = 100


def _roll_window() -> int:
    raw = os.environ.get("OSMIUM_ROLL_WINDOW")
    if raw is None or not str(raw).strip():
        return 30
    try:
        return max(1, int(raw))
    except ValueError:
        return 30


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


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
            sum(o.quantity for o in orders if o.quantity > 0),
            sum(-o.quantity for o in orders if o.quantity < 0),
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
        conversions = 0

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        history: list[list[float]] = store.get("history", [])
        depth = state.order_depths.get(SYMBOL)
        mid = _micro_mid(depth) if depth else None
        if mid is not None:
            history.append([float(state.timestamp), float(mid)])
            if len(history) > HISTORY_CAP:
                history = history[-HISTORY_CAP:]

        if depth is None or not depth.buy_orders or not depth.sell_orders:
            store["history"] = history
            return result, conversions, json.dumps(store)

        mids = [float(h[1]) for h in history]
        rw = _roll_window()
        w = min(len(mids), rw)
        fair_value = sum(mids[-w:]) / float(w)

        position = state.position.get(SYMBOL, 0)
        orders = self.emerald_orders(
            depth,
            fair_value,
            WIDTH,
            position,
            POSITION_LIMIT,
        )
        result[SYMBOL] = orders

        store["history"] = history
        return result, conversions, json.dumps(store)
