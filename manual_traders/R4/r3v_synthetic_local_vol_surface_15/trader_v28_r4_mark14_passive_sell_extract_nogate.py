"""
Same trigger as v27 (Mark 14 seller on aggressive-buy EXTRACT print) **without** Sonic gate.

Ablation to measure how much PnL comes from gate vs counterparty slice alone.
"""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
POS_LIMIT = 200
CLIP = 12
TRIGGER_SELLER = "Mark 14"


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
            if getattr(tr, "seller", None) != TRIGGER_SELLER:
                continue
            if int(getattr(tr, "price", 0)) >= ba:
                fired = True
                break

        if not fired or pos >= POS_LIMIT - CLIP:
            return {}, 0, getattr(state, "traderData", "") or ""

        q = min(CLIP, POS_LIMIT - pos)
        return {U: [Order(U, int(ba) + 1, q)]}, 0, getattr(state, "traderData", "") or ""
