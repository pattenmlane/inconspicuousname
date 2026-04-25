"""
VELVETFRUIT_EXTRACT — simple mean-reversion trader.

Maintains a rolling mid-price mean. Buys when price is below mean,
sells when price is above mean. No bot knowledge used.

Tunable constants:
  WINDOW    — number of ticks to average for the mean (default 50)
  EDGE      — how far from mean before we act (default 1 tick)
"""
from __future__ import annotations

from collections import deque
from datamodel import Order, OrderDepth, TradingState

SYMBOL = "VELVETFRUIT_EXTRACT"
POSITION_LIMIT = 200

WINDOW = 50   # rolling mean window (ticks)
EDGE   = 1    # minimum distance from mean to trade


class Trader:
    def __init__(self):
        self._mid_history: deque[float] = deque(maxlen=WINDOW)

    def run(self, state: TradingState):
        result: dict = {}

        depth: OrderDepth | None = state.order_depths.get(SYMBOL)
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return result, 0, state.traderData or ""

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        mid = (best_bid + best_ask) / 2.0

        self._mid_history.append(mid)

        # Need a full window before acting
        if len(self._mid_history) < WINDOW:
            return result, 0, state.traderData or ""

        mean = sum(self._mid_history) / len(self._mid_history)
        pos  = int(state.position.get(SYMBOL, 0))
        orders: list[Order] = []

        max_buy  = POSITION_LIMIT - pos
        max_sell = POSITION_LIMIT + pos

        # Buy everything below mean - EDGE
        for ask_price in sorted(depth.sell_orders):
            if ask_price < mean - EDGE and max_buy > 0:
                qty = min(abs(depth.sell_orders[ask_price]), max_buy)
                orders.append(Order(SYMBOL, ask_price, qty))
                max_buy -= qty

        # Sell everything above mean + EDGE
        for bid_price in sorted(depth.buy_orders, reverse=True):
            if bid_price > mean + EDGE and max_sell > 0:
                qty = min(abs(depth.buy_orders[bid_price]), max_sell)
                orders.append(Order(SYMBOL, bid_price, -qty))
                max_sell -= qty

        result[SYMBOL] = orders
        return result, 0, state.traderData or ""
