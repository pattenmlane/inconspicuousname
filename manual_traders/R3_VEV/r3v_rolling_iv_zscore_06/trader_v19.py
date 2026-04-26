"""
v19: vouchers_final_strategy — joint tight gate (VEV_5200 & VEV_5300 spread ≤2) + extract only.

Per STRATEGY.txt / ORIGINAL_DISCORD_QUOTES.txt: the empirical edge is on **short-horizon extract
mid** when both legs are tight; voucher legs are mainly the **book-state signal** (Sonic).
This version **does not** trade VEV_5200/5300 for PnL — it only reads their spreads for the gate
and trades **VELVETFRUIT_EXTRACT** on **tight→on** edge (one buy at ask per episode), flatten
on **tight→off** (sell at bid). Ablation: adding 52/53 lifts in the first v19 pass was strongly
negative under worse fills.

Clips are small; hydro untraded.
"""

from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

HYDRO = "HYDROGEL_PACK"
UNDER = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"

TH = 2
MAX_SPREAD_UNDER = 6

LIMITS: dict[str, int] = {
    HYDRO: 200,
    UNDER: 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    V5200: 300,
    V5300: 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

Q_EXTRACT = 8


def _touch(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}

        result: dict[str, list[Order]] = {p: [] for p in LIMITS}
        pos = state.position
        prev_tight = bool(td.get("_prev_tight", 0.0) > 0.5)

        und = state.order_depths.get(UNDER)
        d52 = state.order_depths.get(V5200)
        d53 = state.order_depths.get(V5300)
        if und is None or d52 is None or d53 is None:
            return result, 0, json.dumps(td)

        ub, ua = _touch(und)
        b52, a52 = _touch(d52)
        b53, a53 = _touch(d53)
        if None in (ub, ua, b52, a52, b53, a53):
            return result, 0, json.dumps(td)

        s_und = ua - ub
        s52 = a52 - b52
        s53 = a53 - b53
        joint_tight = s52 <= TH and s53 <= TH

        pu = pos.get(UNDER, 0)
        orders_u: list[Order] = []

        if not joint_tight and prev_tight and pu > 0:
            bb = max(und.buy_orders)
            avail = und.buy_orders.get(bb, 0)
            dq = min(pu, avail)
            if dq > 0:
                orders_u.append(Order(UNDER, bb, -dq))
        elif joint_tight and (not prev_tight) and pu == 0:
            if s_und is not None and s_und <= MAX_SPREAD_UNDER:
                au = min(und.sell_orders)
                avail = abs(und.sell_orders.get(au, 0))
                dq = min(Q_EXTRACT, avail, LIMITS[UNDER])
                if dq > 0:
                    orders_u.append(Order(UNDER, au, dq))

        result[UNDER] = orders_u

        td["_prev_tight"] = 1.0 if joint_tight else 0.0
        td["_s5200"] = float(s52)
        td["_s5300"] = float(s53)
        td["_s_under"] = float(s_und)
        return result, 0, json.dumps(td)
