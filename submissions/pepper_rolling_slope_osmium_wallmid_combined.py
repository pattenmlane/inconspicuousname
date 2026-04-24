from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
# Used only when book has no wall/micro mid and we have no prior tick fair in traderData.
INITIAL_OSMIUM_FAIR = 10_000.0
WIDTH = 2
POSITION_LIMIT = 80
WINDOW = 50
LOG_PREFIX = "OSMIUM_WM_JSON"


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


def _sanitize_slope_history(raw: object, window: int) -> list[list[float]]:
    if not isinstance(raw, list) or window <= 0:
        return []
    out: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            t = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(y):
            out.append([t, y])
    return out[-window * 2 :]


def _mid_pepper(depth: OrderDepth | None) -> float | None:
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None
    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0


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


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid_wall = min(depth.buy_orders.keys())
    ask_wall = max(depth.sell_orders.keys())
    return (bid_wall + ask_wall) / 2.0


def _fair_from_depth(depth: OrderDepth, prev_fair: float | None) -> float:
    w = _wall_mid(depth)
    if w is not None:
        return float(w)
    m = _micro_mid(depth)
    if m is not None:
        return float(m)
    if prev_fair is not None:
        return float(prev_fair)
    return INITIAL_OSMIUM_FAIR


class Trader:
    def _log(self, obj: dict) -> None:
        print(LOG_PREFIX + json.dumps(obj, separators=(",", ":")))

    def emerald_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
        ts: int,
        wall_raw: float | None,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2

        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2

        self._log(
            {
                "t": ts,
                "event": "tick_context",
                "fv": fair_value,
                "wall_raw": wall_raw,
                "pos": position,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bbbf": float(bbbf),
                "baaf": float(baaf),
            }
        )

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    self._log(
                        {
                            "t": ts,
                            "event": "intent",
                            "kind": "take_buy_best_ask_below_fv",
                            "human": f"best_ask={best_ask} < fv={fair_value} → lift ask (buy) qty={quantity}",
                            "price": int(best_ask),
                            "qty": int(quantity),
                        }
                    )
                    orders.append(Order(OSMIUM, int(best_ask), quantity))

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    self._log(
                        {
                            "t": ts,
                            "event": "intent",
                            "kind": "take_sell_best_bid_above_fv",
                            "human": f"best_bid={best_bid} > fv={fair_value} → hit bid (sell) qty={quantity}",
                            "price": int(best_bid),
                            "qty": int(quantity),
                        }
                    )
                    orders.append(Order(OSMIUM, int(best_bid), -quantity))

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            OSMIUM,
            sum([o.quantity for o in orders if o.quantity > 0]),
            sum([-o.quantity for o in orders if o.quantity < 0]),
            fair_value,
            width,
            ts,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            px = int(bbbf + 1)
            self._log(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "passive_mm_buy",
                    "human": f"passive bid (mm) price={px} qty={buy_quantity} (bbbf+1, bbbf={bbbf})",
                    "price": px,
                    "qty": int(buy_quantity),
                }
            )
            orders.append(Order(OSMIUM, px, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            px = int(baaf - 1)
            self._log(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "passive_mm_sell",
                    "human": f"passive ask (mm) price={px} qty={sell_quantity} (baaf-1, baaf={baaf})",
                    "price": px,
                    "qty": int(sell_quantity),
                }
            )
            orders.append(Order(OSMIUM, px, -sell_quantity))

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
        ts: int,
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            self._log(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "clear_long_sell_at_ceil_fv",
                    "human": f"clear long: sell {sent_quantity} @ ceil(fv)={fair_for_ask}",
                    "price": int(fair_for_ask),
                    "qty": int(sent_quantity),
                }
            )
            orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            self._log(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "clear_short_buy_at_floor_fv",
                    "human": f"clear short: buy {sent_quantity} @ floor(fv)={fair_for_bid}",
                    "price": int(fair_for_bid),
                    "qty": int(sent_quantity),
                }
            )
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

        history = _sanitize_slope_history(store.get("history"), WINDOW)

        depth_pe = state.order_depths.get(PEPPER)
        mid_pe = _mid_pepper(depth_pe)
        if mid_pe is not None:
            history.append([float(state.timestamp), float(mid_pe)])
            if len(history) > WINDOW:
                history = history[-WINDOW:]
        store["history"] = history

        pos_pe = int(state.position.get(PEPPER, 0))
        if (
            len(history) >= WINDOW
            and depth_pe is not None
            and depth_pe.buy_orders
            and depth_pe.sell_orders
        ):
            slope = _ols_slope_mid_vs_time(history[-WINDOW:])
            if slope > 0.0:
                target = POSITION_LIMIT
            elif slope < 0.0:
                target = -POSITION_LIMIT
            else:
                target = 0
            diff = target - pos_pe
            if diff > 0 and depth_pe.sell_orders:
                best_ask = min(depth_pe.sell_orders.keys())
                ask_vol = abs(int(depth_pe.sell_orders[best_ask]))
                q = min(diff, POSITION_LIMIT - pos_pe, ask_vol)
                if q > 0:
                    result[PEPPER] = [Order(PEPPER, int(best_ask), int(q))]
            elif diff < 0 and depth_pe.buy_orders:
                best_bid = max(depth_pe.buy_orders.keys())
                bid_vol = int(depth_pe.buy_orders[best_bid])
                q = min(-diff, POSITION_LIMIT + pos_pe, bid_vol)
                if q > 0:
                    result[PEPPER] = [Order(PEPPER, int(best_bid), -int(q))]

        if OSMIUM in state.order_depths:
            depth_os = state.order_depths[OSMIUM]
            wall_raw = _wall_mid(depth_os)
            prev_fair = _store_float(store.get("osmium_last_fair"))
            fair_value = _fair_from_depth(depth_os, prev_fair)
            store["osmium_last_fair"] = fair_value
            pos_os = int(state.position.get(OSMIUM, 0))
            result[OSMIUM] = self.emerald_orders(
                depth_os,
                fair_value,
                WIDTH,
                pos_os,
                POSITION_LIMIT,
                int(state.timestamp),
                wall_raw,
            )

        return result, conversions, json.dumps(store)
