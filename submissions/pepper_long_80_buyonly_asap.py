"""INTARIAN_PEPPER_ROOT — buy only until +80, as fast as the book allows.

Each tick: if position < 80 and there is a best ask, send one aggressive buy
at that ask for ``min(room, displayed ask size)``. Never places a sell.

Backtest (repo root)::

  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt submissions/pepper_long_80_buyonly_asap.py 2-0 \\
  --data Prosperity4Data --match-trades all --no-out
"""

from __future__ import annotations

import json

from datamodel import Order, OrderDepth, TradingState

SYMBOL = "INTARIAN_PEPPER_ROOT"
POSITION_LIMIT = 80


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        trader_data = json.dumps({})

        depth = state.order_depths.get(SYMBOL)
        if depth is None or not depth.sell_orders:
            return result, conversions, trader_data

        pos = int(state.position.get(SYMBOL, 0))
        room = POSITION_LIMIT - pos
        if room <= 0:
            return result, conversions, trader_data

        best_ask = min(depth.sell_orders.keys())
        ask_vol = abs(int(depth.sell_orders[best_ask]))
        qty = min(room, ask_vol)
        if qty <= 0:
            return result, conversions, trader_data

        result[SYMBOL] = [Order(SYMBOL, int(best_ask), int(qty))]
        return result, conversions, trader_data
