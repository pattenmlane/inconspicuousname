"""Strategy C: Take ANY ask <= mid-0.5 and bid >= mid+0.5, then post inside wall.
The Frankfurt 'wm-1 / wm+1' classic."""
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
            wm = (bb + ba) / 2.0
            pos = int(positions.get(sym, 0))
            max_buy = LIM - pos
            max_sell = LIM + pos
            ords = []

            # Take stale quotes
            for sp in sorted(sells.keys()):
                if max_buy <= 0:
                    break
                if sp <= wm - 1:
                    q = min(sells[sp], max_buy)
                    ords.append(Order(sym, sp, q))
                    max_buy -= q
                elif sp <= wm and pos < 0:
                    q = min(sells[sp], max_buy, -pos)
                    if q > 0:
                        ords.append(Order(sym, sp, q))
                        max_buy -= q
            for bp in sorted(buys.keys(), reverse=True):
                if max_sell <= 0:
                    break
                if bp >= wm + 1:
                    q = min(buys[bp], max_sell)
                    ords.append(Order(sym, bp, -q))
                    max_sell -= q
                elif bp >= wm and pos > 0:
                    q = min(buys[bp], max_sell, pos)
                    if q > 0:
                        ords.append(Order(sym, bp, -q))
                        max_sell -= q

            # Make inside wall
            if spread >= 2:
                if spread >= 3:
                    bid_px = bb + 1
                    ask_px = ba - 1
                else:
                    bid_px = bb
                    ask_px = ba
                buy_q = min(SIZE, max_buy)
                sell_q = min(SIZE, max_sell)
                if buy_q > 0:
                    ords.append(Order(sym, bid_px, buy_q))
                if sell_q > 0:
                    ords.append(Order(sym, ask_px, -sell_q))
            if ords:
                result[sym] = ords
        return result, 0, state.traderData or ""
