"""Strategy B: Always quote 1 tick inside the wall (bid+1, ask-1) at max size."""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIM = 300
SIZE = 100


class Trader:
    def run(self, state: TradingState):
        result = {}
        positions = state.position or {}
        for sym in TARGETS:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
            sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
            if not buys or not sells:
                continue
            bb = max(buys); ba = min(sells)
            spread = ba - bb
            if spread < 2:
                continue
            pos = int(positions.get(sym, 0))
            ords = []
            if spread >= 3:
                bid_px = bb + 1
                ask_px = ba - 1
            else:  # spread == 2
                bid_px = bb
                ask_px = ba
            buy_q = min(SIZE, LIM - pos)
            sell_q = min(SIZE, LIM + pos)
            if buy_q > 0:
                ords.append(Order(sym, bid_px, buy_q))
            if sell_q > 0:
                ords.append(Order(sym, ask_px, -sell_q))
            if ords:
                result[sym] = ords
        return result, 0, state.traderData or ""
