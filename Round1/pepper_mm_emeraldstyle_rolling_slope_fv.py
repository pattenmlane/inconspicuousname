"""
INTARIAN_PEPPER_ROOT — same Emerald MM shell as ``pepper_mm_ablation_beta2x.py`` (rank-3
style: takes vs FV, inventory clear, passive ``bbbf+1`` / ``baaf-1``), but **fair value**
comes from a **rolling OLS line** fit to recent micro-mids vs timestamp:

  On each tick with a two-sided book, append ``[timestamp, mid]`` to ``traderData``
  history (capped at ``WINDOW``).

  * If ``len(history) < WINDOW``: ``FV = current micro mid`` (warm-up; same spirit as
    chase-mid until the window fills).
  * Else: OLS ``mid ≈ a + b * t`` on the window; ``FV = a + b * current_timestamp``.

Override window (default **50**, same as ``pepper_rolling_slope_trader``):

  ``PEPPER_ROLL_WINDOW=40`` …

Backtest (from ProsperityRepo):

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/pepper_mm_emeraldstyle_rolling_slope_fv.py" 1 \\
  --data "$PWD/Prosperity4Data" --match-trades worse --no-vis
"""
from __future__ import annotations

import json
import math
import os
from typing import List

from datamodel import Order, OrderDepth, TradingState

SYMBOL = "INTARIAN_PEPPER_ROOT"
WIDTH = 2
POSITION_LIMIT = 80


def _roll_window() -> int:
    raw = os.environ.get("PEPPER_ROLL_WINDOW")
    if raw is None or not str(raw).strip():
        return 50
    try:
        return max(3, int(raw))
    except ValueError:
        return 50


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _ols_intercept_slope(points: list[list[float]]) -> tuple[float, float] | None:
    """OLS y ~ a + b*t. points: [[t, y], ...]. Returns (a, b) or None if singular."""
    n = len(points)
    if n < 2:
        return None
    sum_t = sum_y = sum_tt = sum_ty = 0.0
    for t, y in points:
        sum_t += t
        sum_y += y
        sum_tt += t * t
        sum_ty += t * y
    den = n * sum_tt - sum_t * sum_t
    if den == 0.0:
        return None
    b = (n * sum_ty - sum_t * sum_y) / den
    a = (sum_y - b * sum_t) / n
    return a, b


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
        window = _roll_window()

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        history: list[list[float]] = store.get("history", [])
        if not isinstance(history, list):
            history = []

        depth = state.order_depths.get(SYMBOL)
        mid = _micro_mid(depth) if depth else None

        if mid is not None:
            history.append([float(state.timestamp), float(mid)])
            if len(history) > window:
                history = history[-window:]

        store["history"] = history

        if SYMBOL not in state.order_depths or mid is None:
            return result, 0, json.dumps(store)

        ts = float(state.timestamp)
        if len(history) < window:
            fair_value = float(mid)
        else:
            ab = _ols_intercept_slope(history[-window:])
            if ab is None:
                fair_value = float(mid)
            else:
                a, b = ab
                fair_value = a + b * ts

        position = int(state.position.get(SYMBOL, 0))
        orders = self.emerald_orders(
            state.order_depths[SYMBOL],
            fair_value,
            WIDTH,
            position,
            POSITION_LIMIT,
        )
        result[SYMBOL] = orders
        return result, 0, json.dumps(store)
