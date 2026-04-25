"""Strategy D: Mid-price mean reversion taker.

The lag-1 ACF on d_mid is -0.21 on VEV_5300, -0.14 on VEV_5200, -0.09 on VEV_5100.
After an UP move, mid tends to come back down. So:
  - if last mid > current mid: someone hit a bid; mid is moving down. Buy now (the move will reverse upward).
  - actually the opposite: ACF<0 on changes means a positive change followed by negative.
  Sign convention: r_t = mid_t - mid_{t-1}. ACF(r_t, r_{t-1}) < 0 means after r_{t-1} > 0 we expect r_t < 0.
  So if mid just went UP, expect it to go DOWN next tick. SELL (lift bids).
  If mid just went DOWN, expect it to go UP next tick. BUY (hit asks).

We use a small EMA of mid as the reference. When current mid is N ticks above EMA, sell.
When current mid is N ticks below EMA, buy. Trade size proportional to deviation.
"""
from __future__ import annotations

import json

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIM = 300
EMA_ALPHA = 0.05  # ~20-tick window
THRESH = {"VEV_5100": 1.0, "VEV_5200": 0.7, "VEV_5300": 0.5}  # ticks
SIZE = 80


class Trader:
    def run(self, state: TradingState):
        result = {}
        positions = state.position or {}
        try:
            store = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            store = {}
        emas = store.get("emas", {})

        for sym in TARGETS:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
            sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
            if not buys or not sells:
                continue
            bb = max(buys); ba = min(sells)
            mid = (bb + ba) / 2.0

            ema = emas.get(sym)
            if ema is None:
                emas[sym] = mid
                continue
            new_ema = EMA_ALPHA * mid + (1 - EMA_ALPHA) * ema
            emas[sym] = new_ema

            dev = mid - ema
            thr = THRESH[sym]
            pos = int(positions.get(sym, 0))
            max_buy = LIM - pos
            max_sell = LIM + pos
            ords = []

            # Mid above EMA -> SELL (it'll come back down)
            if dev >= thr and max_sell > 0:
                # Hit the best bid up to deviation/thr scaled size
                size_factor = min(2.0, abs(dev) / thr)
                qty = int(min(buys[bb], max_sell, SIZE * size_factor))
                if qty > 0:
                    ords.append(Order(sym, bb, -qty))
            # Mid below EMA -> BUY
            elif dev <= -thr and max_buy > 0:
                size_factor = min(2.0, abs(dev) / thr)
                qty = int(min(sells[ba], max_buy, SIZE * size_factor))
                if qty > 0:
                    ords.append(Order(sym, ba, qty))

            if ords:
                result[sym] = ords

        store["emas"] = emas
        return result, 0, json.dumps(store, separators=(",", ":"))
