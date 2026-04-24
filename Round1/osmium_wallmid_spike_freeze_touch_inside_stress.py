"""
ASH_COATED_OSMIUM — **Wall-mid spike freeze** (same as
``osmium_mm_emeraldstyle_wallmid_spike_freeze_n.py``) plus **touch inside-spread
stress** from offline stats on day-19 tape:

When the previous tick had a valid **touch** (best bid / best ask) and this tick’s
best bid or best ask sits **strictly inside** that previous spread
(``prev_bid < new_leg < prev_ask``), touch mid tends to jump more on the next steps.
We react in three **causal** ways (no future data):

1. **Mini-freeze** — hold the current fair (post wall-freeze logic) for
   ``OSMIUM_TOUCH_FREEZE_TICKS`` extra trader ticks (counters separate from wall
   freeze; cleared on a **wall** spike).
2. **Wider MM** — add ``OSMIUM_TOUCH_WIDTH_BOOST`` to the Emerald half-spread width
   for that tick when stress fires.
3. **Logging** — ``touch_stress``, ``ask_inside``, ``bid_inside`` on each tick.

Env (optional): ``OSMIUM_WM_SPIKE``, ``OSMIUM_WM_FREEZE_TICKS``,
``OSMIUM_TOUCH_FREEZE_TICKS`` (default **2**), ``OSMIUM_TOUCH_WIDTH_BOOST`` (default **1**).

Backtest::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/osmium_wallmid_spike_freeze_touch_inside_stress.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades all --no-vis
"""
from __future__ import annotations

import json
import math
import os
from typing import List

from datamodel import Order, OrderDepth, TradingState

SYMBOL = "ASH_COATED_OSMIUM"
FALLBACK_FAIR = 10_000.0
WIDTH = 2
POSITION_LIMIT = 80
LOG_PREFIX = "OSMIUM_WMTS_JSON"


def _spike_threshold() -> float:
    raw = os.environ.get("OSMIUM_WM_SPIKE", "3")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 3.0


def _freeze_ticks() -> int:
    raw = os.environ.get("OSMIUM_WM_FREEZE_TICKS", "5")
    try:
        return max(0, int(raw))
    except ValueError:
        return 5


def _touch_freeze_ticks() -> int:
    raw = os.environ.get("OSMIUM_TOUCH_FREEZE_TICKS", "2")
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


def _touch_width_boost() -> int:
    raw = os.environ.get("OSMIUM_TOUCH_WIDTH_BOOST", "1")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


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


def _fair_from_depth_no_store(depth: OrderDepth) -> float:
    w = _wall_mid(depth)
    if w is not None:
        return float(w)
    m = _micro_mid(depth)
    if m is not None:
        return float(m)
    return FALLBACK_FAIR


def _touch_inside_flags(
    depth: OrderDepth,
    prev_bid: float | None,
    prev_ask: float | None,
) -> tuple[bool, bool, bool]:
    """Returns (touch_stress, ask_inside, bid_inside)."""
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

        self._log(
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
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    self._log(
                        {
                            "t": ts,
                            "event": "intent",
                            "kind": "take_buy_best_ask_below_fv",
                            "price": int(best_ask),
                            "qty": int(quantity),
                        }
                    )
                    orders.append(Order(SYMBOL, int(best_ask), quantity))

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
                            "price": int(best_bid),
                            "qty": int(quantity),
                        }
                    )
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
                    "price": px,
                    "qty": int(buy_quantity),
                }
            )
            orders.append(Order(SYMBOL, px, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            px = int(baaf - 1)
            self._log(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "passive_mm_sell",
                    "price": px,
                    "qty": int(sell_quantity),
                }
            )
            orders.append(Order(SYMBOL, px, -sell_quantity))

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
            if sent_quantity > 0:
                self._log(
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
                self._log(
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

        if SYMBOL not in state.order_depths:
            return result, conversions, ""

        try:
            raw_td = state.traderData
            store = json.loads(raw_td) if (raw_td and str(raw_td).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        depth = state.order_depths[SYMBOL]
        wall_raw = _wall_mid(depth)
        spike_thr = _spike_threshold()
        n_freeze = _freeze_ticks()
        n_touch_freeze = _touch_freeze_ticks()
        width_boost = _touch_width_boost()

        prev_pb = store.get("prev_touch_bid")
        prev_pa = store.get("prev_touch_ask")
        prev_bid = float(prev_pb) if prev_pb is not None else None
        prev_ask = float(prev_pa) if prev_pa is not None else None

        touch_stress, ask_inside, bid_inside = _touch_inside_flags(depth, prev_bid, prev_ask)

        prev_wall = store.get("prev_wall")
        last_fv = store.get("last_fv")
        freeze_left = int(store.get("freeze_left", 0) or 0)
        freeze_left_start = freeze_left
        frozen_raw = store.get("frozen_fv")
        frozen_fv: float | None = float(frozen_raw) if frozen_raw is not None else None

        spike_this_tick = (
            wall_raw is not None
            and prev_wall is not None
            and abs(float(wall_raw) - float(prev_wall)) >= spike_thr
        )

        if spike_this_tick and n_freeze > 0:
            fv = float(last_fv) if last_fv is not None else float(wall_raw)
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
            fv = float(last_fv) if last_fv is not None else _fair_from_depth_no_store(depth)
            if freeze_left <= 0:
                frozen_fv = None

        if wall_raw is not None:
            store["prev_wall"] = float(wall_raw)

        ts_fl = int(store.get("ts_freeze_left", 0) or 0)
        ts_fl_start = ts_fl
        ts_fv_raw: float | None = float(store["ts_frozen_fv"]) if store.get("ts_frozen_fv") is not None else None

        if spike_this_tick:
            ts_fl = 0
            ts_fv_raw = None
        elif ts_fl > 0 and ts_fv_raw is not None:
            fv = float(ts_fv_raw)
            ts_fl -= 1
        elif touch_stress and n_touch_freeze > 0 and freeze_left == 0:
            # Do not stack touch mini-freeze on top of an active wall-freeze countdown.
            ts_fv_raw = float(fv)
            ts_fl = n_touch_freeze - 1

        if ts_fv_raw is not None and ts_fl > 0:
            store["ts_frozen_fv"] = float(ts_fv_raw)
            store["ts_freeze_left"] = ts_fl
        else:
            store.pop("ts_frozen_fv", None)
            store["ts_freeze_left"] = 0

        store["last_fv"] = float(fv)
        store["freeze_left"] = freeze_left
        if frozen_fv is not None:
            store["frozen_fv"] = float(frozen_fv)
        else:
            store.pop("frozen_fv", None)

        freeze_active = (
            spike_this_tick
            or (freeze_left_start > 0)
            or (ts_fl_start > 0)
            or (touch_stress and n_touch_freeze > 0)
        )
        fair_value = float(fv)

        bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
        ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if bb is not None and ba is not None:
            store["prev_touch_bid"] = float(bb)
            store["prev_touch_ask"] = float(ba)
        else:
            store.pop("prev_touch_bid", None)
            store.pop("prev_touch_ask", None)

        eff_width = WIDTH + (width_boost if touch_stress else 0)

        position = state.position.get(SYMBOL, 0)
        orders = self.emerald_orders(
            depth,
            fair_value,
            eff_width,
            position,
            POSITION_LIMIT,
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
        result[SYMBOL] = orders

        return result, conversions, json.dumps(store)
