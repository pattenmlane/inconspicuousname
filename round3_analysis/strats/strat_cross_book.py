"""Strategy J: Cross the book on both sides every tick — buy ask, sell bid.

If spread = 2, this loses 2 ticks per round trip. Bad.
If spread = 3 or 4, we lose 3-4 ticks/round trip; but if mid mean-reverts, our
position immediately gains spread/2 in MTM. Worth testing the bookkeeping.

Actually this is GUARANTEED loss = -spread per round-trip. Skip.

Better idea: cross only ONE side per tick, alternating based on a signal.
Use the lag-1 ACF: if mid just went up, the next move is down -> SELL the bid.
"""
from __future__ import annotations

import json

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIM = 300
SIZE = 30


class Trader:

    def run(self, state: TradingState):
        try:
            store = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            store = {}
        last_mids = store.get("last_mid", {})
        positions = state.position or {}
        result = {}

        for sym in TARGETS:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
            sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
            if not buys or not sells:
                continue
            bb = max(buys); ba = min(sells)
            mid = 0.5 * (bb + ba)
            last = last_mids.get(sym, mid)
            last_mids[sym] = mid

            move = mid - last  # >0: up; expect reversal -> SELL the bid
            pos = int(positions.get(sym, 0))
            max_buy = LIM - pos
            max_sell = LIM + pos
            ords = []

            # If mid just went UP enough, sell the bid (expect bounce-back)
            if move >= 1.0 and max_sell > 0:
                qty = min(buys[bb], SIZE, max_sell)
                ords.append(Order(sym, int(bb), -qty))
            # If mid just went DOWN, buy the ask
            elif move <= -1.0 and max_buy > 0:
                qty = min(sells[ba], SIZE, max_buy)
                ords.append(Order(sym, int(ba), qty))

            # Inventory unwind toward 0 when no signal
            if not ords and pos != 0:
                if pos > 0 and max_sell > 0:
                    qty = min(buys[bb], pos, max_sell, SIZE)
                    if qty > 0: ords.append(Order(sym, int(bb), -qty))
                elif pos < 0 and max_buy > 0:
                    qty = min(sells[ba], -pos, max_buy, SIZE)
                    if qty > 0: ords.append(Order(sym, int(ba), qty))

            if ords:
                result[sym] = ords

        store["last_mid"] = last_mids
        return result, 0, json.dumps(store, separators=(",", ":"))
