"""
ASH_COATED_OSMIUM — Wall-based Emerald MM with **asymmetric take edges** and **no 10k**.

Fair value: **wall mid**, else **micro mid**, else **last tick FV** from ``traderData`` (no
constant anchor such as 10_000).

Takes (asymmetric vs vanilla Emerald lift/hit):
  * Buy lift: ``best_ask < fv - BUY_EDGE``   (default **2**)
  * Sell hit: ``best_bid > fv + SELL_EDGE`` (default **3**)

Edges are **fixed** every tick (env / defaults only) — no tick-count or day-length switching.

Passive ladder and inventory clear unchanged from ``osmium_mm_emeraldstyle_wallmid``.

Override edges with env (integers):
  ``OSMIUM_BUY_EDGE=1 OSMIUM_SELL_EDGE=2 python3 -m prosperity4bt ...``
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
LOG_PREFIX = "OSMIUM_WTE_JSON"


def _edges() -> tuple[int, int]:
    be = os.environ.get("OSMIUM_BUY_EDGE", "2")
    se = os.environ.get("OSMIUM_SELL_EDGE", "3")
    try:
        return max(0, int(be)), max(0, int(se))
    except ValueError:
        return 2, 3


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
    def _log(self, obj: dict) -> None:
        print(LOG_PREFIX + json.dumps(obj, separators=(",", ":")))

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
        ts: int,
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

    def emerald_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
        ts: int,
        wall_raw: float | None,
        buy_e: int,
        sell_e: int,
    ) -> List[Order]:
        orders: List[Order] = []

        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2
        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2

        self._log(
            {
                "t": ts,
                "fv": fair_value,
                "wall": wall_raw,
                "buy_e": buy_e,
                "sell_e": sell_e,
                "pos": position,
            }
        )

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -int(order_depth.sell_orders[best_ask])
            if best_ask < fair_value - buy_e and best_ask_amount > 0:
                q = min(best_ask_amount, position_limit - position)
                if q > 0:
                    orders.append(Order(SYMBOL, int(best_ask), q))

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = int(order_depth.buy_orders[best_bid])
            if best_bid > fair_value + sell_e and best_bid_amount > 0:
                q = min(best_bid_amount, position_limit + position)
                if q > 0:
                    orders.append(Order(SYMBOL, int(best_bid), -q))

        buy_vol = sum(o.quantity for o in orders if o.quantity > 0)
        sell_vol = sum(-o.quantity for o in orders if o.quantity < 0)

        bov, sov = self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            SYMBOL,
            buy_vol,
            sell_vol,
            fair_value,
            width,
            ts,
        )

        buy_q = position_limit - (position + bov)
        if buy_q > 0:
            orders.append(Order(SYMBOL, int(bbbf + 1), int(buy_q)))
        sell_q = position_limit + (position - sov)
        if sell_q > 0:
            orders.append(Order(SYMBOL, int(baaf - 1), -int(sell_q)))

        return orders

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        try:
            store = json.loads(state.traderData) if (state.traderData and str(state.traderData).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}
        store.pop("n_ticks", None)

        last = store.get("last_fv")
        if isinstance(last, (int, float)):
            last_f = float(last)
        else:
            last_f = None

        if SYMBOL not in state.order_depths:
            return result, conversions, json.dumps(store)

        depth = state.order_depths[SYMBOL]
        fv = _fair(depth)
        if fv is None:
            if last_f is None:
                return result, conversions, json.dumps(store)
            fv = last_f
        else:
            store["last_fv"] = float(fv)

        buy_e, sell_e = _edges()
        wr = _wall_mid(depth)
        pos = int(state.position.get(SYMBOL, 0))
        result[SYMBOL] = self.emerald_orders(
            depth, fv, WIDTH, pos, POSITION_LIMIT, int(state.timestamp), wr, buy_e, sell_e
        )
        return result, conversions, json.dumps(store)
