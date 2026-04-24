"""Pepper-only R2 submission leg with **PEPPER_TARGET_LONG = 75** (80 cap).

Same logic as ``r2_submission_pepper_only.py`` / ``round2work/r2_submission.py``
drift Emerald MM + slope safeguard; only ``PEPPER_TARGET_LONG`` differs from the
default submission (70 → 75, i.e. 75/5 vs 70/10 headroom).
"""

from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

BETA_DRIFT = 1.0e-3
PEPPER = "INTARIAN_PEPPER_ROOT"
WIDTH = 2

PEPPER_TARGET_LONG = 75
PEPPER_POSITION_LIMIT = 80

PEPPER_SLOPE_WINDOW = 50
PEPPER_SLOPE_SAFEGUARD = -0.00015
PEPPER_SLOPE_HIST = "pepper_slope_hist"

MAF_BID_XIREC = 6001


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


def _book_volume_int(v: object) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _parse_trader_data(raw: object) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}
    s = str(raw).strip() if raw else ""
    if not s:
        return {}
    try:
        out = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _position_int(state: TradingState, sym: str) -> int:
    try:
        v = state.position.get(sym, 0)
        return int(v)
    except (TypeError, ValueError):
        return 0


def _timestamp_int(state: TradingState) -> int:
    try:
        return int(state.timestamp)
    except (TypeError, ValueError, AttributeError):
        return 0


def _finite_or(x: float, fallback: float) -> float:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return fallback


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _sanitize_pepper_slope_hist(raw: object, window: int) -> list[list[float]]:
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


class Trader:
    def bid(self) -> int:
        return MAF_BID_XIREC

    def pepper_emerald_orders(
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

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * _book_volume_int(order_depth.sell_orders[best_ask])
            cap_room = position_limit - position
            need = max(0, target_long - position)
            if best_ask < fair_value:
                quantity = min(best_ask_amount, cap_room)
            else:
                quantity = min(best_ask_amount, cap_room, need) if need > 0 else 0
            if quantity > 0:
                orders.append(Order(PEPPER, int(best_ask), quantity))

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
                    orders.append(Order(PEPPER, int(best_bid), -quantity))

        buy_order_volume, sell_order_volume = self.pepper_clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            PEPPER,
            sum([o.quantity for o in orders if o.quantity > 0]),
            sum([-o.quantity for o in orders if o.quantity < 0]),
            fair_value,
            width,
            target_long,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(PEPPER, int(bbbf + 1), buy_quantity))

        pos_after = position + buy_order_volume - sell_order_volume
        max_passive_sell = max(0, pos_after - target_long)
        std_sell = position_limit + (position - sell_order_volume)
        sell_quantity = min(std_sell, max_passive_sell)
        if sell_quantity > 0:
            orders.append(Order(PEPPER, int(baaf - 1), -sell_quantity))

        return orders

    def pepper_clear_position_order(
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
            clear_quantity = min(abs(_book_volume_int(order_depth.sell_orders[fair_for_bid])), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def _pepper_crash_reversal_orders(
        self, order_depth: OrderDepth, position: int, position_limit: int
    ) -> List[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        target = -position_limit
        diff = target - position
        if diff < 0:
            best_bid = max(order_depth.buy_orders.keys())
            bid_vol = _book_volume_int(order_depth.buy_orders[best_bid])
            q = min(-diff, position_limit + position, bid_vol)
            if q > 0:
                return [Order(PEPPER, int(best_bid), -int(q))]
        if diff > 0:
            best_ask = min(order_depth.sell_orders.keys())
            ask_vol = abs(_book_volume_int(order_depth.sell_orders[best_ask]))
            q = min(diff, position_limit - position, ask_vol)
            if q > 0:
                return [Order(PEPPER, int(best_ask), int(q))]
        return []

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        store = _parse_trader_data(getattr(state, "traderData", None))

        depths = getattr(state, "order_depths", None)
        if not isinstance(depths, dict):
            depths = {}

        target_long = _target_long()
        position_limit_pe = _position_limit()
        alpha = _store_float(store.get("alpha"))
        depth_pe = depths.get(PEPPER)
        mid_pe = _micro_mid(depth_pe) if depth_pe else None

        hist = _sanitize_pepper_slope_hist(store.get(PEPPER_SLOPE_HIST), PEPPER_SLOPE_WINDOW)
        ts = _timestamp_int(state)
        if mid_pe is not None:
            hist.append([float(ts), float(mid_pe)])
            if len(hist) > PEPPER_SLOPE_WINDOW * 2:
                hist = hist[-(PEPPER_SLOPE_WINDOW * 2) :]
        store[PEPPER_SLOPE_HIST] = hist

        slope_crash = False
        if (
            len(hist) >= PEPPER_SLOPE_WINDOW
            and depth_pe is not None
            and depth_pe.buy_orders
            and depth_pe.sell_orders
        ):
            slope = _ols_slope_mid_vs_time(hist[-PEPPER_SLOPE_WINDOW :])
            slope_crash = slope < PEPPER_SLOPE_SAFEGUARD

        if slope_crash and PEPPER in depths:
            pos_pe = _position_int(state, PEPPER)
            result[PEPPER] = self._pepper_crash_reversal_orders(
                depths[PEPPER], pos_pe, position_limit_pe
            )
        else:
            if alpha is None and mid_pe is not None:
                alpha = float(mid_pe) - BETA_DRIFT * float(ts)
                store["alpha"] = alpha

            if alpha is not None and PEPPER in depths:
                drifted = float(alpha) + BETA_DRIFT * float(ts)
                fair_pe = _finite_or(drifted, float(alpha))
                pos_pe = _position_int(state, PEPPER)
                result[PEPPER] = self.pepper_emerald_orders(
                    depths[PEPPER],
                    fair_pe,
                    WIDTH,
                    pos_pe,
                    position_limit_pe,
                    target_long,
                )

        try:
            td_out = json.dumps(store, separators=(",", ":"))
        except (TypeError, ValueError):
            td_out = "{}"
        return result, conversions, td_out
