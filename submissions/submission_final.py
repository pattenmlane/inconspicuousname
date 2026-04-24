"""
Round 1 — **submission final** (pepper 70/10 drift MM + plain wall-mid osmium).

* **INTARIAN_PEPPER_ROOT** — same as ``pepper_70_10_osmium_spike_freeze_combined.py``
  pepper leg: drift FV ``alpha + BETA_DRIFT * timestamp``, Emerald MM, target long 70 /
  10-lot buffer to 80, plus a **rolling 50-tick touch-mid slope** safeguard that pushes
  inventory short if the local trend breaks below ``PEPPER_SLOPE_SAFEGUARD``.

* **ASH_COATED_OSMIUM** — same as ``submissions/osmium_mm_emeraldstyle_wallmid.py``:
  Emerald MM with FV = wall mid → micro mid → **10_000** (no spike freeze).

``traderData`` JSON: ``alpha`` and ``pepper_slope_hist`` (pepper). Osmium leg is stateless aside from alpha.

No stdlib **environment** module (submit rules). Tune pepper targets via module constants.

Backtest::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/submissions/submission_final.py" 1 \\
  --data "$PWD/Prosperity4Data" --match-trades worse --no-vis
"""
from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

BETA_DRIFT = 1.0e-3
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
WIDTH = 2

PEPPER_TARGET_LONG = 70
PEPPER_POSITION_LIMIT = 80

PEPPER_SLOPE_WINDOW = 50
PEPPER_SLOPE_SAFEGUARD = -0.00015
PEPPER_SLOPE_HIST = "pepper_slope_hist"

OSMIUM_POSITION_LIMIT = 80
FALLBACK_OSMIUM = 10_000.0
LOG_OSMIUM = "OSMIUM_WM_JSON"


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


def _wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid_wall = min(depth.buy_orders.keys())
    ask_wall = max(depth.sell_orders.keys())
    return (bid_wall + ask_wall) / 2.0


def _fair_from_depth(depth: OrderDepth) -> float:
    w = _wall_mid(depth)
    if w is not None:
        return float(w)
    m = _micro_mid(depth)
    if m is not None:
        return float(m)
    return FALLBACK_OSMIUM


class Trader:
    def _log_osm(self, obj: dict) -> None:
        print(LOG_OSMIUM + json.dumps(obj, separators=(",", ":")))

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
            best_ask_amount = -1 * int(order_depth.sell_orders[best_ask])
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
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
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
            bid_vol = int(order_depth.buy_orders[best_bid])
            q = min(-diff, position_limit + position, bid_vol)
            if q > 0:
                return [Order(PEPPER, int(best_bid), -int(q))]
        if diff > 0:
            best_ask = min(order_depth.sell_orders.keys())
            ask_vol = abs(int(order_depth.sell_orders[best_ask]))
            q = min(diff, position_limit - position, ask_vol)
            if q > 0:
                return [Order(PEPPER, int(best_ask), int(q))]
        return []

    def osmium_emerald_orders(
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

        self._log_osm(
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
                    self._log_osm(
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
                    self._log_osm(
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
                    "human": f"passive bid (mm) price={px} qty={buy_quantity} (bbbf+1, bbbf={bbbf})",
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
                    "human": f"passive ask (mm) price={px} qty={sell_quantity} (baaf-1, baaf={baaf})",
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
            if sent_quantity > 0:
                self._log_osm(
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

        target_long = _target_long()
        position_limit_pe = _position_limit()
        alpha = _store_float(store.get("alpha"))
        depth_pe = state.order_depths.get(PEPPER)
        mid_pe = _micro_mid(depth_pe) if depth_pe else None

        hist = _sanitize_pepper_slope_hist(store.get(PEPPER_SLOPE_HIST), PEPPER_SLOPE_WINDOW)
        if mid_pe is not None:
            hist.append([float(state.timestamp), float(mid_pe)])
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
            slope = _ols_slope_mid_vs_time(hist[-PEPPER_SLOPE_WINDOW:])
            slope_crash = slope < PEPPER_SLOPE_SAFEGUARD

        if slope_crash and PEPPER in state.order_depths:
            pos_pe = int(state.position.get(PEPPER, 0))
            result[PEPPER] = self._pepper_crash_reversal_orders(
                state.order_depths[PEPPER], pos_pe, position_limit_pe
            )
        else:
            if alpha is None and mid_pe is not None:
                alpha = float(mid_pe) - BETA_DRIFT * float(state.timestamp)
                store["alpha"] = alpha

            if alpha is not None and PEPPER in state.order_depths:
                fair_pe = float(alpha) + BETA_DRIFT * float(state.timestamp)
                pos_pe = int(state.position.get(PEPPER, 0))
                result[PEPPER] = self.pepper_emerald_orders(
                    state.order_depths[PEPPER],
                    fair_pe,
                    WIDTH,
                    pos_pe,
                    position_limit_pe,
                    target_long,
                )

        if OSMIUM in state.order_depths:
            depth_os = state.order_depths[OSMIUM]
            wall_raw = _wall_mid(depth_os)
            fair_value = _fair_from_depth(depth_os)
            pos_os = int(state.position.get(OSMIUM, 0))
            result[OSMIUM] = self.osmium_emerald_orders(
                depth_os,
                fair_value,
                WIDTH,
                pos_os,
                OSMIUM_POSITION_LIMIT,
                int(state.timestamp),
                wall_raw,
            )

        return result, conversions, json.dumps(store)
