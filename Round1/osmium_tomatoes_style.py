"""
ASH_COATED_OSMIUM — **TOMATOES_v1** logic (wall mid + take + one-tick overbid/undercut MM).

Ported from ``TOMATOES/TOMATOES_v1_tutorialstats_web=1706_all=18132_worse=18278_none=1068.py``:
same ``ProductTrader`` behavior, only ``SYMBOL`` and docstring changed. Position limit 80
matches P4 constants for osmium.

Backtest (from repo root, adjust ``--data`` if needed):

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/osmium_tomatoes_style.py" 1--2 1--1 1-0 1-19 \\
  --data "$PWD/Prosperity4Data" --match-trades worse --no-vis
"""
from __future__ import annotations

import json

from datamodel import Order, OrderDepth, TradingState

SYMBOL = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80


class ProductTrader:
    def __init__(self, state, prints, new_trader_data):
        self.orders = []
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data

        self.position_limit = POSITION_LIMIT
        self.initial_position = self.state.position.get(SYMBOL, 0)
        self.max_allowed_buy_volume = self.position_limit - self.initial_position
        self.max_allowed_sell_volume = self.position_limit + self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()

    def get_order_depth(self):
        order_depth, buy_orders, sell_orders = {}, {}, {}
        try:
            order_depth = self.state.order_depths[SYMBOL]
        except Exception:
            pass
        try:
            buy_orders = {
                bp: abs(bv)
                for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            }
        except Exception:
            pass
        try:
            sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except Exception:
            pass
        return buy_orders, sell_orders

    def get_walls(self):
        bid_wall = wall_mid = ask_wall = None
        try:
            bid_wall = min([x for x, _ in self.mkt_buy_orders.items()])
        except Exception:
            pass
        try:
            ask_wall = max([x for x, _ in self.mkt_sell_orders.items()])
        except Exception:
            pass
        try:
            wall_mid = (bid_wall + ask_wall) / 2
        except Exception:
            pass
        return bid_wall, wall_mid, ask_wall

    def bid(self, price, volume):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        if abs_volume <= 0:
            return
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(Order(SYMBOL, int(price), abs_volume))

    def ask(self, price, volume):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        if abs_volume <= 0:
            return
        self.max_allowed_sell_volume -= abs_volume
        self.orders.append(Order(SYMBOL, int(price), -abs_volume))

    def get_orders(self):
        if self.wall_mid is None:
            return {SYMBOL: self.orders}

        # Take favorable asks and flatten shorts at fair.
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= self.wall_mid - 1:
                self.bid(sp, sv)
            elif sp <= self.wall_mid and self.initial_position < 0:
                self.bid(sp, min(sv, abs(self.initial_position)))

        # Take favorable bids and flatten longs at fair.
        for bp, bv in self.mkt_buy_orders.items():
            if bp >= self.wall_mid + 1:
                self.ask(bp, bv)
            elif bp >= self.wall_mid and self.initial_position > 0:
                self.ask(bp, min(bv, self.initial_position))

        # Provide around fair with one-tick overbid/undercut where possible.
        bid_price = int(self.bid_wall + 1)
        ask_price = int(self.ask_wall - 1)

        for bp, bv in self.mkt_buy_orders.items():
            overbidding_price = bp + 1
            if bv > 1 and overbidding_price < self.wall_mid:
                bid_price = max(bid_price, overbidding_price)
                break
            elif bp < self.wall_mid:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in self.mkt_sell_orders.items():
            underbidding_price = sp - 1
            if sv > 1 and underbidding_price > self.wall_mid:
                ask_price = min(ask_price, underbidding_price)
                break
            elif sp > self.wall_mid:
                ask_price = min(ask_price, sp)
                break

        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)
        return {SYMBOL: self.orders}


class Trader:
    def run(self, state: TradingState):
        result = {}
        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position,
            },
        }

        def export(prints_):
            try:
                print(json.dumps(prints_))
            except Exception:
                pass

        conversions = 0
        if SYMBOL in state.order_depths:
            try:
                trader = ProductTrader(state, prints, new_trader_data)
                result.update(trader.get_orders())
            except Exception:
                pass

        try:
            final_trader_data = json.dumps(new_trader_data)
        except Exception:
            final_trader_data = ""

        export(prints)
        return result, conversions, final_trader_data
