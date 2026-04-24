"""
INTARIAN_PEPPER_ROOT — drift FV Emerald MM with **target long 70** and **10-lot spread buffer**.

Same fair value as ``pepper_mm_emeraldstyle_drift.py`` (``alpha + BETA_DRIFT * timestamp``).

Inventory intent:
  * Try to stay **at least +70** long (soft floor): take-sells, clear-sells, and passive
    asks are **capped** so we do not push position below 70.
  * Use **positions 70..80** as headroom for normal Emerald spread farming (the top
    **10** units vs the 70 “core”).
  * If position **< 70** and the book has asks, **lift best ask** up to ``min(ask size,
    room to 70)`` in addition to the usual MM orders (refill toward target).

``POSITION_LIMIT`` stays **80** (exchange cap). Edit ``PEPPER_TARGET_LONG`` /
``PEPPER_POSITION_LIMIT`` in this file (no stdlib **environment** module on submit):

  ``PEPPER_TARGET_LONG=70 PEPPER_POSITION_LIMIT=80`` …

  Literal **80/0** (peg long at cap, no sell-down buffer): ``PEPPER_TARGET_LONG=80`` with
  ``PEPPER_POSITION_LIMIT=80`` — take-sells / passive sells / clear sells that would drop
  position below **80** are capped to zero; only buys refill toward **+80``.

Backtest:

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/pepper_mm_emeraldstyle_drift_long70_buffer10.py" 1 \\
  --data "$PWD/Prosperity4Data" --match-trades worse --no-vis
"""
from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

BETA_DRIFT = 1.0e-3
SYMBOL = "INTARIAN_PEPPER_ROOT"
WIDTH = 2

PEPPER_TARGET_LONG = 70
PEPPER_POSITION_LIMIT = 80


def _target_long() -> int:
    lim = _position_limit()
    return max(0, min(PEPPER_TARGET_LONG, lim))


def _position_limit() -> int:
    return max(1, PEPPER_POSITION_LIMIT)


def _store_float(x: object) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


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
        target_long: int,
    ) -> List[Order]:
        orders: List[Order] = []

        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2

        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2

        # One lift at best ask: refill toward target when under 70; if ask is cheap vs FV, lift up to cap.
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * int(order_depth.sell_orders[best_ask])
            cap_room = position_limit - position
            need = max(0, target_long - position)
            if best_ask < fair_value:
                quantity = min(best_ask_amount, cap_room)
            else:
                quantity = min(best_ask_amount, cap_room, need) if need > 0 else 0
            if quantity > 0:
                orders.append(Order(SYMBOL, int(best_ask), quantity))

        buy_vol_takes = sum(o.quantity for o in orders if o.quantity > 0)
        sell_vol_takes = sum(-o.quantity for o in orders if o.quantity < 0)
        pos_after_buys = position + buy_vol_takes

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = int(order_depth.buy_orders[best_bid])
            if best_bid > fair_value:
                max_sell = max(0, pos_after_buys - target_long)
                quantity = min(best_bid_amount, position_limit + position, max_sell)
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
            target_long,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(SYMBOL, int(bbbf + 1), buy_quantity))

        pos_after = position + buy_order_volume - sell_order_volume
        max_passive_sell = max(0, pos_after - target_long)
        std_sell = position_limit + (position - sell_order_volume)
        sell_quantity = min(std_sell, max_passive_sell)
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
        target_long: int,
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            cap = max(0, position_after_take - target_long)
            sent_quantity = min(sell_quantity, clear_quantity, cap)
            if sent_quantity > 0:
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
        target_long = _target_long()
        position_limit = _position_limit()

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        alpha = _store_float(store.get("alpha"))
        depth = state.order_depths.get(SYMBOL)
        mid = _micro_mid(depth) if depth else None

        if alpha is None and mid is not None:
            alpha = float(mid) - BETA_DRIFT * float(state.timestamp)
            store["alpha"] = alpha

        if alpha is None or SYMBOL not in state.order_depths:
            return result, 0, json.dumps(store)

        fair_value = float(alpha) + BETA_DRIFT * float(state.timestamp)
        position = int(state.position.get(SYMBOL, 0))
        orders = self.emerald_orders(
            state.order_depths[SYMBOL],
            fair_value,
            WIDTH,
            position,
            position_limit,
            target_long,
        )
        result[SYMBOL] = orders
        return result, 0, json.dumps(store)
