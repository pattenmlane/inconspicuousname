"""
Round 4 Phase-3 **Mark 01 → Mark 22** on **VEV_5300** (Phase-1 pair), **aggressive sell** only.

Tape fact (r4_p1_trades_enriched + gate merge): **all** 132 such prints in days 1–3 occur at
timestamps with joint Sonic gate **tight** (no loose stratum) — so gate×pair **interaction
Welch (tight vs loose) is inestimable**; short-horizon **fwd** markouts in Phase-1 are **negative**
at K=5/20 for this print type.

This probe: when gate tight AND a tape print shows **buyer=Mark 01**, **seller=Mark 22** on
`VEV_5300` with **price <= bid1** (aggr_sell), **short** clip at **bb-1** for `--match-trades worse`
fills. Tests sign agreement with **negative** tape fwd (fade / lean short, not a blind basket follow).

Position limit: 300 per VEV (same as round4description).
"""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

VEV = "VEV_5300"
V5200 = "VEV_5200"
GATE5300 = "VEV_5300"
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
    s2 = _spread(depths.get(GATE5300))
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

        pos = int((getattr(state, "position", {}) or {}).get(VEV, 0))
        mkt = getattr(state, "market_trades", None) or {}
        if not isinstance(mkt, dict):
            mkt = {}

        depth_v = depths.get(VEV)
        bb, ba = _bb_ba(depth_v)
        if bb is None or ba is None:
            return {}, 0, getattr(state, "traderData", "") or ""

        fired = False
        for tr in mkt.get(VEV, []):
            if getattr(tr, "buyer", None) != "Mark 01":
                continue
            if getattr(tr, "seller", None) != "Mark 22":
                continue
            if int(getattr(tr, "price", 0)) > bb:
                continue
            fired = True
            break

        if not fired:
            return {}, 0, getattr(state, "traderData", "") or ""

        room = pos + POS_LIM
        if room <= 0:
            return {}, 0, getattr(state, "traderData", "") or ""

        q = min(CLIP, room)
        px = max(int(bb) - 1, 0)
        return {VEV: [Order(VEV, px, -q)]}, 0, getattr(state, "traderData", "") or ""
