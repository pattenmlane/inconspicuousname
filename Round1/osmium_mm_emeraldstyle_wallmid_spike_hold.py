"""
ASH_COATED_OSMIUM — same Emerald MM as ``osmium_mm_emeraldstyle_wallmid.py``,
but when **raw wall mid** moves by at least ``SPIKE`` from the previous tick’s
raw wall mid, **fair value stays at the previous tick’s FV** (sticky on jumps).

Raw wall mid: min(bid px) + max(ask px) over the book, divided by 2 (same as
Hedgehogs). First tick or missing book: same fallbacks as ``wallmid`` script.

Override spike size: ``OSMIUM_WM_SPIKE=4`` (default **3**).

State in ``traderData`` JSON: ``prev_wall``, ``last_fv``.

**Logging:** ``OSMIUM_WMSH_JSON``.

Backtest:
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/osmium_mm_emeraldstyle_wallmid_spike_hold.py" 1--2 \\
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
LOG_PREFIX = "OSMIUM_WMSH_JSON"


def _spike_threshold() -> float:
    raw = os.environ.get("OSMIUM_WM_SPIKE", "3")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 3.0


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
        spike_hold: bool,
        prev_wall: float | None,
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
                "spike_hold": spike_hold,
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
                            "human": f"best_bid={best_bid} > fv={fair_value} → hit bid (sell) qty={quantity}",
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
                    "human": f"passive bid (mm) price={px} qty={buy_quantity} (bbbf+1, bbbf={bbbf})",
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
                    "human": f"passive ask (mm) price={px} qty={sell_quantity} (baaf-1, baaf={baaf})",
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

        prev_wall = store.get("prev_wall")
        last_fv = store.get("last_fv")

        if wall_raw is not None:
            target_fv = float(wall_raw)
            spike_hold = False
            if prev_wall is not None and abs(float(wall_raw) - float(prev_wall)) >= spike_thr:
                if last_fv is not None:
                    target_fv = float(last_fv)
                    spike_hold = True
            store["prev_wall"] = float(wall_raw)
            store["last_fv"] = float(target_fv)
        else:
            spike_hold = False
            if last_fv is not None:
                target_fv = float(last_fv)
            else:
                target_fv = _fair_from_depth_no_store(depth)
                store["last_fv"] = float(target_fv)

        fair_value = float(target_fv)

        position = state.position.get(SYMBOL, 0)
        orders = self.emerald_orders(
            depth,
            fair_value,
            WIDTH,
            position,
            POSITION_LIMIT,
            int(state.timestamp),
            wall_raw,
            spike_hold,
            prev_wall,
        )
        result[SYMBOL] = orders

        return result, conversions, json.dumps(store)
