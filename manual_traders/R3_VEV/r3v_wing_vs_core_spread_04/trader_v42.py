"""
v42 — vouchers_final_strategy/ only: joint 5200+5300 BBO spread ≤2 gate.

- On rising edge (tight, was wide last step): one-way add to small long VELVETFRUIT_EXTRACT
  (up to ENTRY_LONG) — optional STRATEGY “short-hold / favorable mid” layer.
- While wide or after leaving tight: work down any long in clips (Sonic: wide = execution noise).

No voucher MM, no hydrogel. traderData: prev_tight for edge detect.
"""
from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
H = "HYDROGEL_PACK"

LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in VOUCHERS},
}

TIGHT_S5200_S5300_TH = 2
ENTRY_LONG = 8
MAX_CLOSE_CLIP = 20


def _mid(depth: OrderDepth) -> tuple[float, int, int] | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    return (bb + ba) / 2.0, bb, ba


def _spread_bbo(depth: OrderDepth) -> int | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return int(min(depth.sell_orders) - max(depth.buy_orders))


def _joint_tight(
    d52: OrderDepth | None, d53: OrderDepth | None
) -> bool:
    if d52 is None or d53 is None:
        return False
    s5, s3 = _spread_bbo(d52), _spread_bbo(d53)
    if s5 is None or s3 is None:
        return False
    return s5 <= TIGHT_S5200_S5300_TH and s3 <= TIGHT_S5200_S5300_TH


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        orders: dict[str, list[Order]] = {p: [] for p in LIMITS}
        d52 = state.order_depths.get(VEV_5200)
        d53 = state.order_depths.get(VEV_5300)
        du = state.order_depths.get(U)

        if du is None or not _mid(du):
            return orders, 0, json.dumps(td)

        tight = _joint_tight(d52, d53)
        prev = bool(td.get("prev_tight", False))
        td["prev_tight"] = tight

        pu = int(state.position.get(U, 0))
        u_lim = LIMITS[U]

        if tight and (not prev) and pu < ENTRY_LONG and du.sell_orders:
            q = min(ENTRY_LONG - max(0, pu), u_lim - pu)
            if q > 0:
                orders[U].append(Order(U, int(min(du.sell_orders)), int(q)))

        if (not tight) and pu > 0 and du.buy_orders:
            q = min(pu, MAX_CLOSE_CLIP)
            orders[U].append(Order(U, int(max(du.buy_orders)), -q))

        return orders, 0, json.dumps(td)
