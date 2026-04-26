"""
Round 4 — Phase-2-informed **tape** strategy (counterparty IDs).

When `state.market_trades` contains a **VELVETFRUIT_EXTRACT** print this tick with
**buyer == Mark 67** and price at/through the ask (aggressive buy vs current book),
buy a small clip **one tick above best ask** (works with `--match-trades worse`).

Requires backtester to populate `state.market_trades` from tape (see test_runner change).
No HYDROGEL / VEV orders in v1 (extract-only probe from Phase 1 edge).
"""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

U = "VELVETFRUIT_EXTRACT"
POS_LIMIT = 200
CLIP = 12
TRIGGER_BUYER = "Mark 67"


def _bb_ba(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None, None
    bb = max(depth.buy_orders.keys())
    ba = min(abs(p) for p in depth.sell_orders.keys())
    if ba <= bb:
        return None, None
    return int(bb), int(ba)


class Trader:
    def run(self, state: TradingState):
        depths = getattr(state, "order_depths", {}) or {}
        pos = int((getattr(state, "position", {}) or {}).get(U, 0))
        mkt = getattr(state, "market_trades", None) or {}
        if not isinstance(mkt, dict):
            mkt = {}

        depth_u = depths.get(U)
        bb, ba = _bb_ba(depth_u)
        if bb is None or ba is None:
            return {}, 0, getattr(state, "traderData", "") or ""

        fired = False
        for tr in mkt.get(U, []):
            if getattr(tr, "buyer", None) != TRIGGER_BUYER:
                continue
            px = int(getattr(tr, "price", 0))
            if px >= ba:
                fired = True
                break

        if not fired or pos >= POS_LIMIT - CLIP:
            return {}, 0, getattr(state, "traderData", "") or ""

        q = min(CLIP, POS_LIMIT - pos)
        if q <= 0:
            return {}, 0, getattr(state, "traderData", "") or ""
        return {U: [Order(U, int(ba) + 1, q)]}, 0, getattr(state, "traderData", "") or ""
