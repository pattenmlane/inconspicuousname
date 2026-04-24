"""
INTARIAN_PEPPER_ROOT — **rolling OLS slope (50 mids)** picks the regime; **drift FV**
Emerald MM farms spread with a **70 / 10** inventory band in-direction.

Same drift fair value as ``pepper_mm_emeraldstyle_drift.py`` (``alpha + BETA_DRIFT * t``).

Regimes (once ``len(history) >= WINDOW`` from ``[timestamp, micro_mid]`` in ``traderData``):

* **slope > 0 (bull):** same 70/10 **long** rules as ``pepper_mm_emeraldstyle_drift_long70_buffer10.py``
  (floor +70, cap +80, capped sells / refill buys).
* **slope < 0 (bear):** symmetric **−70 / −10** short band (stay at most **−70**, use **−80..−70**
  for MM; capped buys / refill sells). Mirror of the long logic.
* **slope == 0 (flat):** vanilla Emerald drift MM (no inventory band).

Warm-up (``len(history) < WINDOW``): neutral drift MM so we quote while the slope warms.

Env:

* ``PEPPER_ROLL_WINDOW`` (default **50**)
* ``PEPPER_TARGET_MAG`` — half-band magnitude (default **70** → long +70/+80, short −70/−80)

Backtest:

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/pepper_mm_rolling_slope_signal_drift_long70_buffer10.py" 1 \\
  --data "$PWD/Prosperity4Data" --match-trades worse --no-vis
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Literal

from datamodel import Order, OrderDepth, TradingState

BETA_DRIFT = 1.0e-3
SYMBOL = "INTARIAN_PEPPER_ROOT"
WIDTH = 2


def _roll_window() -> int:
    raw = os.environ.get("PEPPER_ROLL_WINDOW", "50")
    try:
        return max(3, int(raw))
    except ValueError:
        return 50


def _target_mag() -> int:
    raw = os.environ.get("PEPPER_TARGET_MAG", "70")
    try:
        return max(1, int(raw))
    except ValueError:
        return 70


def _position_limit() -> int:
    raw = os.environ.get("PEPPER_POSITION_LIMIT", "80")
    try:
        return max(1, int(raw))
    except ValueError:
        return 80


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _ols_slope_mid_vs_time(points: list[list[float]]) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    sum_t = sum_y = sum_tt = sum_ty = 0.0
    for t, y in points:
        sum_t += t
        sum_y += y
        sum_tt += t * t
        sum_ty += t * y
    den = n * sum_tt - sum_t * sum_t
    if den == 0.0:
        return 0.0
    return (n * sum_ty - sum_t * sum_y) / den


Regime = Literal["bull", "bear", "neutral"]


class Trader:
    def clear_neutral(
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
    ) -> tuple[int, int]:
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

    def clear_bull(
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
    ) -> tuple[int, int]:
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

    def clear_bear(
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
        target_short: int,
    ) -> tuple[int, int]:
        """target_short is negative (e.g. -70). Cap cover-buys so position stays <= -70."""
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
            cap = max(0, target_short - position_after_take)
            sent_quantity = min(buy_quantity, clear_quantity, cap)
            if sent_quantity > 0:
                orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def emerald_neutral(
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
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * int(order_depth.sell_orders[best_ask])
            if best_ask < fair_value and best_ask_amount > 0:
                q = min(best_ask_amount, position_limit - position)
                if q > 0:
                    orders.append(Order(SYMBOL, int(best_ask), q))
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = int(order_depth.buy_orders[best_bid])
            if best_bid > fair_value and best_bid_amount > 0:
                q = min(best_bid_amount, position_limit + position)
                if q > 0:
                    orders.append(Order(SYMBOL, int(best_bid), -q))
        bov, sov = self.clear_neutral(
            orders, order_depth, position, position_limit, SYMBOL,
            sum(o.quantity for o in orders if o.quantity > 0),
            sum(-o.quantity for o in orders if o.quantity < 0),
            fair_value, width,
        )
        buy_q = position_limit - (position + bov)
        if buy_q > 0:
            orders.append(Order(SYMBOL, int(bbbf + 1), buy_q))
        sell_q = position_limit + (position - sov)
        if sell_q > 0:
            orders.append(Order(SYMBOL, int(baaf - 1), -sell_q))
        return orders

    def emerald_bull(
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
        if order_depth.sell_orders:
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
        pos_after_buys = position + buy_vol_takes
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = int(order_depth.buy_orders[best_bid])
            if best_bid > fair_value:
                max_sell = max(0, pos_after_buys - target_long)
                quantity = min(best_bid_amount, position_limit + position, max_sell)
                if quantity > 0:
                    orders.append(Order(SYMBOL, int(best_bid), -quantity))
        bov, sov = self.clear_bull(
            orders, order_depth, position, position_limit, SYMBOL,
            sum(o.quantity for o in orders if o.quantity > 0),
            sum(-o.quantity for o in orders if o.quantity < 0),
            fair_value, width, target_long,
        )
        buy_q = position_limit - (position + bov)
        if buy_q > 0:
            orders.append(Order(SYMBOL, int(bbbf + 1), buy_q))
        pos_after = position + bov - sov
        max_ps = max(0, pos_after - target_long)
        sell_q = min(position_limit + (position - sov), max_ps)
        if sell_q > 0:
            orders.append(Order(SYMBOL, int(baaf - 1), -sell_q))
        return orders

    def emerald_bear(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
        target_short: int,
    ) -> List[Order]:
        """target_short is -70. Band [-80, -70]."""
        orders: List[Order] = []
        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2
        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2

        # Refill sell at bid when not short enough (position > -70)
        if order_depth.buy_orders and position > target_short:
            best_bid = max(order_depth.buy_orders.keys())
            bid_vol = int(order_depth.buy_orders[best_bid])
            room = position_limit + position
            need = position - target_short
            if best_bid > fair_value:
                quantity = min(bid_vol, room)
            else:
                quantity = min(bid_vol, room, need) if need > 0 else 0
            if quantity > 0:
                orders.append(Order(SYMBOL, int(best_bid), -int(quantity)))

        sell_vol_r = sum(-o.quantity for o in orders if o.quantity < 0)
        pos_after_sells = position - sell_vol_r

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = int(order_depth.buy_orders[best_bid])
            if best_bid > fair_value:
                floor_short = -position_limit
                max_sell = max(0, pos_after_sells - floor_short)
                quantity = min(best_bid_amount, position_limit + position, max_sell)
                if quantity > 0:
                    orders.append(Order(SYMBOL, int(best_bid), -quantity))

        pos_after_takes = position + sum(o.quantity for o in orders if o.quantity > 0) - sum(
            -o.quantity for o in orders if o.quantity < 0
        )

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * int(order_depth.sell_orders[best_ask])
            if best_ask < fair_value:
                max_buy = max(0, target_short - pos_after_takes)
                quantity = min(best_ask_amount, position_limit - position, max_buy)
                if quantity > 0:
                    orders.append(Order(SYMBOL, int(best_ask), quantity))

        bov, sov = self.clear_bear(
            orders, order_depth, position, position_limit, SYMBOL,
            sum(o.quantity for o in orders if o.quantity > 0),
            sum(-o.quantity for o in orders if o.quantity < 0),
            fair_value, width, target_short,
        )

        sell_q = position_limit + (position - sov)
        if sell_q > 0:
            orders.append(Order(SYMBOL, int(baaf - 1), -sell_q))
        pos_after = position + bov - sov
        max_pb = max(0, target_short - pos_after)
        buy_q_std = position_limit - (position + bov)
        buy_q = min(buy_q_std, max_pb)
        if buy_q > 0:
            orders.append(Order(SYMBOL, int(bbbf + 1), buy_q))
        return orders

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        window = _roll_window()
        tm = _target_mag()
        lim = _position_limit()
        target_long = min(tm, lim)
        target_short = max(-tm, -lim)

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

        alpha = store.get("alpha")
        if alpha is None and mid is not None:
            alpha = float(mid) - BETA_DRIFT * float(state.timestamp)
            store["alpha"] = alpha

        if alpha is None or SYMBOL not in state.order_depths or mid is None:
            return result, 0, json.dumps(store)

        fair_value = float(alpha) + BETA_DRIFT * float(state.timestamp)
        position = int(state.position.get(SYMBOL, 0))
        d = state.order_depths[SYMBOL]

        regime: Regime = "neutral"
        if len(history) >= window:
            slope = _ols_slope_mid_vs_time(history[-window:])
            if slope > 0.0:
                regime = "bull"
            elif slope < 0.0:
                regime = "bear"
            else:
                regime = "neutral"

        if regime == "bull":
            result[SYMBOL] = self.emerald_bull(d, fair_value, WIDTH, position, lim, target_long)
        elif regime == "bear":
            result[SYMBOL] = self.emerald_bear(d, fair_value, WIDTH, position, lim, target_short)
        else:
            result[SYMBOL] = self.emerald_neutral(d, fair_value, WIDTH, position, lim)

        return result, 0, json.dumps(store)
