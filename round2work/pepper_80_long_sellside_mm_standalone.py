"""
Pepper: target **+80** with **no headroom reserved for MM**.

- While ``position < 80``: same aggressive lift-at-best-ask logic as ``pepper_80_0_long_only_standalone``.
- While ``position >= 80``: **sell-side only** Emerald (drift fair): take-sells, clear at ceil(fv),
  passive **ask** at ``baaf-1`` — **no** passive bids and **no** lift-at-ask takes, so inventory is
  not pushed above 80 by MM. Passive ask size includes a small clip so we still quote at the cap.

Fair value: same anchored linear drift as 70/10 (``alpha`` + ``BETA_DRIFT`` × time).

Compare: ``compare_pepper_70_10_vs_80_0_backtest.py`` (extend to three strategies).
"""

from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

BETA_DRIFT = 1.0e-3
PEPPER = "INTARIAN_PEPPER_ROOT"
WIDTH = 2

PEPPER_POSITION_LIMIT = 80
TARGET_LONG = 80

# Extra passive ask size (lots) when at/under target so we still MM at ~80 long.
MM_PASSIVE_SELL_CLIP = 10


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


def _aggressive_toward_80(depth: OrderDepth, position: int) -> List[Order]:
    if not depth.buy_orders or not depth.sell_orders:
        return []
    lim = _position_limit()
    tgt = min(TARGET_LONG, lim)
    need = tgt - position
    if need <= 0:
        return []
    best_ask = min(depth.sell_orders.keys())
    ask_vol = abs(int(depth.sell_orders[best_ask]))
    q = min(need, lim - position, ask_vol)
    if q <= 0:
        return []
    return [Order(PEPPER, int(best_ask), int(q))]


class Trader:
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

    def pepper_sell_side_mm_only(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
        target_long: int,
    ) -> List[Order]:
        """Emerald on sell side only: no lift-at-ask, no passive bid."""
        orders: List[Order] = []

        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2

        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2
        _ = bbbf  # no passive bid; keep for symmetry if extended later

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = int(order_depth.buy_orders[best_bid])
            if best_bid > fair_value:
                max_sell = max(0, position - target_long)
                quantity = min(best_bid_amount, position_limit + position, max_sell)
                if quantity > 0:
                    orders.append(Order(PEPPER, int(best_bid), -quantity))

        buy_order_volume = sum(o.quantity for o in orders if o.quantity > 0)
        sell_order_volume = sum(-o.quantity for o in orders if o.quantity < 0)

        buy_order_volume, sell_order_volume = self.pepper_clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            PEPPER,
            buy_order_volume,
            sell_order_volume,
            fair_value,
            width,
            target_long,
        )

        pos_after = position + buy_order_volume - sell_order_volume
        excess = max(0, pos_after - target_long)
        clip = min(MM_PASSIVE_SELL_CLIP, max(0, pos_after))
        max_passive_sell = excess + clip
        std_sell = position_limit + (position - sell_order_volume)
        sell_quantity = min(std_sell, max_passive_sell)
        if sell_quantity > 0:
            orders.append(Order(PEPPER, int(baaf - 1), -sell_quantity))

        return orders

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        position_limit = _position_limit()
        alpha = _store_float(store.get("alpha"))
        depth_pe = state.order_depths.get(PEPPER)
        mid_pe = _micro_mid(depth_pe) if depth_pe else None

        if alpha is None and mid_pe is not None:
            alpha = float(mid_pe) - BETA_DRIFT * float(state.timestamp)
            store["alpha"] = alpha

        if depth_pe is None or alpha is None or PEPPER not in state.order_depths:
            return result, conversions, json.dumps(store)

        fair_pe = float(alpha) + BETA_DRIFT * float(state.timestamp)
        pos_pe = int(state.position.get(PEPPER, 0))

        if pos_pe < TARGET_LONG:
            result[PEPPER] = _aggressive_toward_80(state.order_depths[PEPPER], pos_pe)
        else:
            result[PEPPER] = self.pepper_sell_side_mm_only(
                state.order_depths[PEPPER],
                fair_pe,
                WIDTH,
                pos_pe,
                position_limit,
                TARGET_LONG,
            )

        return result, conversions, json.dumps(store)
