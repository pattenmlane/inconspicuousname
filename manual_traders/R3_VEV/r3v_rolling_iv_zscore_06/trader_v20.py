"""
v20: same thesis as v19, plus K-step extract **mid momentum** (STRATEGY.txt: K=20 forward).

Track rolling extract mids in `traderData` (same time-scale as the analysis script). On
**tight-gate turn-on** (loose → joint tight), go long touch **only** if
`mid(t) - mid(t-20) > 0` (or small threshold) — aligns with the published positive mean
forward Δmid under tight *when* the local drift is not strongly against you.

On gate turn-off, flatten. Hydro untraded; 5200/5300 for gate only.
"""

from __future__ import annotations

import json
from collections import deque
from datamodel import Order, OrderDepth, TradingState

HYDRO = "HYDROGEL_PACK"
UNDER = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"

TH = 2
K_MOM = 20
MOM_MIN = 0.0
MAX_SPREAD_UNDER = 6
Q_EXTRACT = 8

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

        mhist: deque[float] = deque(
            (float(x) for x in td.get("_m_under", [])), maxlen=K_MOM + 1
        )
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

        s_mid = 0.5 * (ub + ua)
        mhist.append(s_mid)
        s_und = ua - ub
        s52 = a52 - b52
        s53 = a53 - b53
        joint_tight = s52 <= TH and s53 <= TH

        mom = 0.0
        if len(mhist) == K_MOM + 1:
            mom = s_mid - mhist[0]

        pu = pos.get(UNDER, 0)
        orders_u: list[Order] = []

        if not joint_tight and prev_tight and pu > 0:
            bb = max(und.buy_orders)
            avail = und.buy_orders.get(bb, 0)
            dq = min(pu, avail)
            if dq > 0:
                orders_u.append(Order(UNDER, bb, -dq))
        elif joint_tight and (not prev_tight) and pu == 0:
            if (
                s_und is not None
                and s_und <= MAX_SPREAD_UNDER
                and len(mhist) > K_MOM
                and mom > MOM_MIN
            ):
                au = min(und.sell_orders)
                avail = abs(und.sell_orders.get(au, 0))
                dq = min(Q_EXTRACT, avail, LIMITS[UNDER])
                if dq > 0:
                    orders_u.append(Order(UNDER, au, dq))

        result[UNDER] = orders_u

        td["_m_under"] = list(mhist)
        td["_prev_tight"] = 1.0 if joint_tight else 0.0
        td["_s5200"] = float(s52)
        td["_s5300"] = float(s53)
        td["_s_under"] = float(s_und)
        td["_mom_k20"] = mom
        return result, 0, json.dumps(td)
