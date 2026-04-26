"""
Round 4 — **Mark 49** **seller** on **aggressive-buy** **VELVETFRUIT_EXTRACT** (passive at ask),
**only** under Sonic joint gate.

Phase-1 participant row: seller Mark49 × aggr_buy × K=20, n=104, mean fwd20 **~+1.83** (t~5).
Merged gate: **tight** n=15 mean **~+1.17** vs **loose** n=89 mean **~+1.94** — Welch **p≈0.43**
(gate does **not** improve this slice; day-1 tight n=2 mean negative — unstable).

Live probe: buy clip **ba+1** (same as v4), CLIP=8, for `--match-trades worse`.
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
POS_LIM = 200
CLIP = 8


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

        depth = depths.get(U)
        bb, ba = _bb_ba(depth)
        if bb is None or ba is None:
            return {}, 0, getattr(state, "traderData", "") or ""

        fired = False
        for tr in mkt.get(U, []):
            if getattr(tr, "seller", None) != "Mark 49":
                continue
            if int(getattr(tr, "price", 0)) < ba:
                continue
            fired = True
            break

        if not fired or pos >= POS_LIM - CLIP:
            return {}, 0, getattr(state, "traderData", "") or ""

        q = min(CLIP, POS_LIM - pos)
        return {U: [Order(U, int(ba) + 1, q)]}, 0, getattr(state, "traderData", "") or ""
