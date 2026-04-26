"""Same as v31 but **no orders when state.backtest_day==3** (attribution / stress)."""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

SYM = "VEV_4000"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
TH = 2
POS_LIM = 300
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
        if int(getattr(state, "backtest_day", -1)) == 3:
            return {}, 0, getattr(state, "traderData", "") or ""

        depths = getattr(state, "order_depths", {}) or {}
        if not _joint_tight(depths):
            return {}, 0, getattr(state, "traderData", "") or ""

        pos = int((getattr(state, "position", {}) or {}).get(SYM, 0))
        mkt = getattr(state, "market_trades", None) or {}
        if not isinstance(mkt, dict):
            mkt = {}

        depth = depths.get(SYM)
        bb, ba = _bb_ba(depth)
        if bb is None or ba is None:
            return {}, 0, getattr(state, "traderData", "") or ""

        fired = False
        for tr in mkt.get(SYM, []):
            if getattr(tr, "buyer", None) != "Mark 14":
                continue
            if getattr(tr, "seller", None) != "Mark 38":
                continue
            if int(getattr(tr, "price", 0)) > bb:
                continue
            fired = True
            break

        if not fired or pos >= POS_LIM - CLIP:
            return {}, 0, getattr(state, "traderData", "") or ""

        q = min(CLIP, POS_LIM - pos)
        return {SYM: [Order(SYM, int(ba) + 1, q)]}, 0, getattr(state, "traderData", "") or ""
