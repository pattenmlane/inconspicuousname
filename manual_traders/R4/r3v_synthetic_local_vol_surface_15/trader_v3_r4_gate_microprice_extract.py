"""
Round 4 — **Sonic joint gate** + **Phase-2 microprice** filter (inclineGod: book state).

When VEV_5200 and VEV_5300 spreads are both ≤2, compute extract **microprice**:
  (bid1 * ask_vol1 + ask1 * bid_vol1) / (bid_vol1 + ask_vol1)
If microprice > mid + **MICRO_TH** (upward book pressure; Phase-2 lag-gap correlated with next Δmid),
buy VELVETFRUIT_EXTRACT at **ba+1** (small clip).

No counterparty ID required — stacks Sonic gate with microstructure from r4_p2_microprice_summary.
"""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
TH = 2
POS_LIMIT = 200
CLIP = 10
MICRO_TH = 0.35


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


def _micro_vs_mid(depth: OrderDepth | None) -> tuple[float, float, int, int] | None:
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders.keys())
    ba = min(abs(p) for p in depth.sell_orders.keys())
    bv = int(depth.buy_orders.get(bb, 0))
    av = int(abs(depth.sell_orders.get(ba, 0)))
    if bv + av <= 0 or ba <= bb:
        return None
    mid = 0.5 * (bb + ba)
    micro = (bb * av + ba * bv) / float(bv + av)
    return micro, mid, bb, ba


class Trader:
    def run(self, state: TradingState):
        depths = getattr(state, "order_depths", {}) or {}
        if not _joint_tight(depths):
            return {}, 0, getattr(state, "traderData", "") or ""

        pos = int((getattr(state, "position", {}) or {}).get(U, 0))
        m = _micro_vs_mid(depths.get(U))
        if m is None:
            return {}, 0, getattr(state, "traderData", "") or ""
        micro, mid, bb, ba = m
        if micro <= mid + MICRO_TH:
            return {}, 0, getattr(state, "traderData", "") or ""

        if pos >= POS_LIMIT - CLIP:
            return {}, 0, getattr(state, "traderData", "") or ""
        q = min(CLIP, POS_LIMIT - pos)
        return {U: [Order(U, int(ba) + 1, q)]}, 0, getattr(state, "traderData", "") or ""
