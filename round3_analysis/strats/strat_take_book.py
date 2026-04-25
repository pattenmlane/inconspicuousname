"""Strategy A: Take the entire L1 book on both sides every tick on 5100/5200/5300.

Theory: there's a "stale book" effect — if I take the L1 ask at tick t, on tick t+1
the bid often moves up past my purchase price, so I can sell back to the new bid.
This monetizes the bid-ask bounce captured by the lag-1 ACF (-0.1 to -0.2 on these strikes).
"""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIM = 300


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
            pos = int(positions.get(sym, 0))
            max_buy = LIM - pos
            max_sell = LIM + pos
            ords = []
            # Sweep all asks
            for sp in sorted(sells.keys()):
                if max_buy <= 0:
                    break
                q = min(sells[sp], max_buy)
                ords.append(Order(sym, sp, q))
                max_buy -= q
            # Sweep all bids
            for bp in sorted(buys.keys(), reverse=True):
                if max_sell <= 0:
                    break
                q = min(buys[bp], max_sell)
                ords.append(Order(sym, bp, -q))
                max_sell -= q
            if ords:
                result[sym] = ords
        return result, 0, state.traderData or ""
