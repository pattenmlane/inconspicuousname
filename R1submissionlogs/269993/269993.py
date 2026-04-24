from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
WIDTH = 2
POSITION_LIMIT = 80
WINDOW = 50

OSMIUM_POSITION_LIMIT = 80
LOG_OSMIUM = "OSMIUM_SR1_JSON"

OSMIUM_WM_SPIKE = 3.0
OSMIUM_WM_FREEZE_TICKS = 5
OSMIUM_TOUCH_FREEZE_TICKS = 2
OSMIUM_TOUCH_WIDTH_BOOST = 1


def _store_float(x: object) -> float | None:
    """Parse JSON-stored numbers safely (reject bool / non-finite)."""
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


def _store_nonneg_int(x: object, default: int = 0) -> int:
    if isinstance(x, bool):
        return default
    if isinstance(x, int):
        return max(0, x)
    if isinstance(x, float) and math.isfinite(x):
        return max(0, int(x))
    if isinstance(x, str):
        try:
            v = float(x)
            if math.isfinite(v):
                return max(0, int(v))
        except ValueError:
            pass
    return default


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
    return out[-window:]


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


def _spike_threshold() -> float:
    return max(0.0, float(OSMIUM_WM_SPIKE))


def _freeze_ticks() -> int:
    return max(0, int(OSMIUM_WM_FREEZE_TICKS))


def _touch_freeze_ticks() -> int:
    return max(0, int(OSMIUM_TOUCH_FREEZE_TICKS))


def _touch_width_boost() -> int:
    return max(0, int(OSMIUM_TOUCH_WIDTH_BOOST))


def _wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid_wall = min(depth.buy_orders.keys())
    ask_wall = max(depth.sell_orders.keys())
    return (bid_wall + ask_wall) / 2.0


def _touch_inside_flags(
    depth: OrderDepth,
    prev_bid: float | None,
    prev_ask: float | None,
) -> tuple[bool, bool, bool]:
    if prev_bid is None or prev_ask is None:
        return False, False, False
    if not depth.buy_orders or not depth.sell_orders:
        return False, False, False
    bb = float(max(depth.buy_orders.keys()))
    ba = float(min(depth.sell_orders.keys()))
    ask_in = prev_bid < ba < prev_ask
    bid_in = prev_bid < bb < prev_ask
    return ask_in or bid_in, ask_in, bid_in


class Trader:
    def _log_osm(self, obj: dict) -> None:
        print(LOG_OSMIUM + json.dumps(obj, separators=(",", ":")))

    def osmium_emerald_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
        ts: int,
        wall_raw: float | None,
        freeze_active: bool,
        freeze_left_start: int,
        freeze_left_end: int,
        prev_wall: float | None,
        spike_this_tick: bool,
        touch_stress: bool,
        ask_inside: bool,
        bid_inside: bool,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2

        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2

        self._log_osm(
            {
                "t": ts,
                "event": "tick_context",
                "fv": fair_value,
                "wall_raw": wall_raw,
                "prev_wall": prev_wall,
                "freeze_active": freeze_active,
                "freeze_left_start": freeze_left_start,
                "freeze_left_end": freeze_left_end,
                "spike_this_tick": spike_this_tick,
                "touch_stress": touch_stress,
                "ask_inside": ask_inside,
                "bid_inside": bid_inside,
                "pos": position,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bbbf": float(bbbf),
                "baaf": float(baaf),
                "width": width,
            }
        )

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * int(order_depth.sell_orders[best_ask])
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    self._log_osm(
                        {
                            "t": ts,
                            "event": "intent",
                            "kind": "take_buy_best_ask_below_fv",
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
                    self._log_osm(
                        {
                            "t": ts,
                            "event": "intent",
                            "kind": "take_sell_best_bid_above_fv",
                            "price": int(best_bid),
                            "qty": int(quantity),
                        }
                    )
                    orders.append(Order(OSMIUM, int(best_bid), -quantity))

        buy_order_volume, sell_order_volume = self.osmium_clear_position_order(
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
            self._log_osm(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "passive_mm_buy",
                    "price": px,
                    "qty": int(buy_quantity),
                }
            )
            orders.append(Order(OSMIUM, px, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            px = int(baaf - 1)
            self._log_osm(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "passive_mm_sell",
                    "price": px,
                    "qty": int(sell_quantity),
                }
            )
            orders.append(Order(OSMIUM, px, -sell_quantity))

        return orders

    def osmium_clear_position_order(
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
            if sent_quantity > 0:
                self._log_osm(
                    {
                        "t": ts,
                        "event": "intent",
                        "kind": "clear_long_sell_at_ceil_fv",
                        "price": int(fair_for_ask),
                        "qty": int(sent_quantity),
                    }
                )
                orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                self._log_osm(
                    {
                        "t": ts,
                        "event": "intent",
                        "kind": "clear_short_buy_at_floor_fv",
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

        history = _sanitize_slope_history(store.get("history"), WINDOW * 2)

        depth_pe = state.order_depths.get(PEPPER)
        mid_pe = _mid_pepper(depth_pe)
        if mid_pe is not None:
            history.append([float(state.timestamp), float(mid_pe)])
            if len(history) > WINDOW:
                history = history[-WINDOW:]
        store["history"] = history

        pos_pe = int(state.position.get(PEPPER, 0))
        if len(history) >= WINDOW and depth_pe is not None and depth_pe.buy_orders and depth_pe.sell_orders:
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
            spike_thr = _spike_threshold()
            n_freeze = _freeze_ticks()
            n_touch_freeze = _touch_freeze_ticks()
            width_boost = _touch_width_boost()

            prev_pb = store.get("osm_prev_touch_bid")
            prev_pa = store.get("osm_prev_touch_ask")
            prev_touch_bid = _store_float(prev_pb)
            prev_touch_ask = _store_float(prev_pa)

            touch_stress, ask_inside, bid_inside = _touch_inside_flags(
                depth_os, prev_touch_bid, prev_touch_ask
            )

            prev_wall = _store_float(store.get("osm_prev_wall"))
            last_fv = _store_float(store.get("osm_last_fv"))
            last_wall_mid = _store_float(store.get("osm_last_wall_mid"))
            freeze_left = _store_nonneg_int(store.get("osm_freeze_left"), 0)
            freeze_left_start = freeze_left
            frozen_fv = _store_float(store.get("osm_frozen_fv"))
            fv: float | None

            spike_this_tick = (
                wall_raw is not None
                and prev_wall is not None
                and abs(float(wall_raw) - float(prev_wall)) >= spike_thr
            )

            if spike_this_tick and n_freeze > 0:
                lf = last_fv
                fv = lf if lf is not None else float(wall_raw)
                frozen_fv = fv
                freeze_left = n_freeze - 1
            elif freeze_left > 0 and frozen_fv is not None:
                fv = frozen_fv
                freeze_left -= 1
            elif wall_raw is not None:
                fv = float(wall_raw)
                freeze_left = 0
                frozen_fv = None
            else:
                lf = last_fv
                lw = last_wall_mid
                fv = lf if lf is not None else lw
                if freeze_left <= 0:
                    frozen_fv = None

            if wall_raw is not None:
                store["osm_prev_wall"] = float(wall_raw)
                store["osm_last_wall_mid"] = float(wall_raw)

            ts_fl = _store_nonneg_int(store.get("osm_ts_freeze_left"), 0)
            ts_fl_start = ts_fl
            ts_fv_raw = _store_float(store.get("osm_ts_frozen_fv"))

            if spike_this_tick:
                ts_fl = 0
                ts_fv_raw = None
            elif ts_fl > 0 and ts_fv_raw is not None:
                fv = float(ts_fv_raw)
                ts_fl -= 1
            elif touch_stress and n_touch_freeze > 0 and freeze_left == 0 and fv is not None:
                ts_fv_raw = float(fv)
                ts_fl = n_touch_freeze - 1

            if ts_fv_raw is not None and ts_fl > 0:
                store["osm_ts_frozen_fv"] = float(ts_fv_raw)
                store["osm_ts_freeze_left"] = ts_fl
            else:
                store.pop("osm_ts_frozen_fv", None)
                store["osm_ts_freeze_left"] = 0

            if fv is not None:
                store["osm_last_fv"] = float(fv)
            store["osm_freeze_left"] = freeze_left
            if frozen_fv is not None:
                store["osm_frozen_fv"] = float(frozen_fv)
            else:
                store.pop("osm_frozen_fv", None)

            freeze_active = (
                spike_this_tick
                or (freeze_left_start > 0)
                or (ts_fl_start > 0)
                or (touch_stress and n_touch_freeze > 0)
            )

            if depth_os.buy_orders and depth_os.sell_orders:
                store["osm_prev_touch_bid"] = float(max(depth_os.buy_orders.keys()))
                store["osm_prev_touch_ask"] = float(min(depth_os.sell_orders.keys()))
            else:
                store.pop("osm_prev_touch_bid", None)
                store.pop("osm_prev_touch_ask", None)

            eff_width = WIDTH + (width_boost if touch_stress else 0)

            pos_os = int(state.position.get(OSMIUM, 0))
            if fv is not None:
                fair_value = float(fv)
                result[OSMIUM] = self.osmium_emerald_orders(
                    depth_os,
                    fair_value,
                    eff_width,
                    pos_os,
                    OSMIUM_POSITION_LIMIT,
                    int(state.timestamp),
                    wall_raw,
                    freeze_active,
                    freeze_left_start,
                    freeze_left,
                    prev_wall,
                    spike_this_tick,
                    touch_stress,
                    ask_inside,
                    bid_inside,
                )

        return result, conversions, json.dumps(store)