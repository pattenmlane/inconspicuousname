"""
Round 1 smoke test: keep buying INTARIAN_PEPPER_ROOT until position reaches +80.

Each timestamp, if position < 80 and the book has asks, send an aggressive buy
at best ask (size capped by remaining room and top-of-book size). Repeats until flat at limit.

Run (from ProsperityRepo root), e.g. round 1 day -2:
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/pepper_buy_hold_t0_80.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades all --no-vis
"""
from __future__ import annotations

import json

from datamodel import Order, TradingState

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
