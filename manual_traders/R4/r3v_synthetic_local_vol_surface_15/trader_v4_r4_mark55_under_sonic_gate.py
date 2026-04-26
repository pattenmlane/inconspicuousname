"""
Round 4 — **Mark 55** aggressive buy on **VELVETFRUIT_EXTRACT** under **Sonic joint gate** (same as v2 for Mark 67).

Phase-1/2: Mark 55 aggr_buy extract had positive mean fwd20 with larger n than Mark67;
Phase-3 extension: tight vs loose Welch on merged trades was not significant (p≈0.52) — this
trader tests live PnL when gating that slice only.
"""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

U = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
TH = 2
POS_LIMIT = 200
CLIP = 12
TRIGGER_BUYER = "Mark 55"


def _spread(depth: OrderDepth | None) -> int | None:
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders.keys())
    ba = min(abs(p) for p in depth.sell_orders.keys())
    if ba <= bb:
        return None
    return int(ba - bb)


def _joint_tight(depths: dict) -> bool:
    s1 = _spread(depths.get(V5200))
    s2 = _spread(depths.get(V5300))
    if s1 is None or s2 is None:
        return False
    return s1 <= TH and s2 <= TH


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
        if not _joint_tight(depths):
            return {}, 0, getattr(state, "traderData", "") or ""

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
            if int(getattr(tr, "price", 0)) >= ba:
                fired = True
                break

        if not fired or pos >= POS_LIMIT - CLIP:
            return {}, 0, getattr(state, "traderData", "") or ""

        q = min(CLIP, POS_LIMIT - pos)
        return {U: [Order(U, int(ba) + 1, q)]}, 0, getattr(state, "traderData", "") or ""
