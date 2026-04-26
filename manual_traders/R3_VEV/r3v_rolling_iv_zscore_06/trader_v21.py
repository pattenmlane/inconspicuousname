"""
v21: vouchers_final_strategy — joint s5200/s5300 <= 2 (Sonic) + two-sided **quotes** on
VELVETFRUIT_EXTRACT, VEV_5200, VEV_5300 when the gate is on (tight-surface MM, not mid taker).

Gate on: if spread >= 3 use bid+1 / ask-1; if spread 1-2 quote at best bid/ask. Gate off: flatten
inventory in those three. Q/POS_SOFT tune. No hydrogel.
"""

from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

HYDRO = "HYDROGEL_PACK"
UNDER = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"

TH = 2
MAX_SPREAD_UNDER = 8
MAX_SPREAD_VEV = 6
# When joint tight, 5200/5300 have sp \u2208 {1,2} usually \u2014 must quote **at touch** (or better when sp\u22653)
Q = 20
POS_SOFT = 100

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


def _mm_orders(
    sym: str,
    depth: OrderDepth,
    pos: int,
    cap_spread: int,
) -> list[Order]:
    ub, ua = _touch(depth)
    if ub is None or ua is None:
        return []
    sp = ua - ub
    if sp > cap_spread or sp < 1:
        return []
    if sp >= 3:
        bid_px, ask_px = ub + 1, ua - 1
    else:
        # sp 1 or 2: at touch (only way to two-side quote without crossing)
        bid_px, ask_px = ub, ua
    if bid_px >= ask_px:
        return []
    lim = LIMITS[sym]
    o: list[Order] = []
    q_buy = min(Q, max(0, lim - pos))
    q_sell = min(Q, max(0, lim + pos))
    if pos > POS_SOFT and q_sell:
        o.append(Order(sym, ask_px, -q_sell))
    elif pos < -POS_SOFT and q_buy:
        o.append(Order(sym, bid_px, q_buy))
    else:
        if q_buy:
            o.append(Order(sym, bid_px, q_buy))
        if q_sell:
            o.append(Order(sym, ask_px, -q_sell))
    return o


def _flatten_long(sym: str, depth: OrderDepth, pos: int) -> list[Order]:
    if pos <= 0:
        return []
    bb = max(depth.buy_orders)
    avail = depth.buy_orders.get(bb, 0)
    q = min(pos, max(0, avail), 80)
    return [Order(sym, bb, -q)] if q else []


def _flatten_short(sym: str, depth: OrderDepth, pos: int) -> list[Order]:
    if pos >= 0:
        return []
    aa = min(depth.sell_orders)
    avail = abs(depth.sell_orders.get(aa, 0))
    q = min(-pos, max(0, avail), 80)
    return [Order(sym, aa, q)] if q else []


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}

        result: dict[str, list[Order]] = {p: [] for p in LIMITS}
        pos = state.position

        und = state.order_depths.get(UNDER)
        d52 = state.order_depths.get(V5200)
        d53 = state.order_depths.get(V5300)
        if und is None or d52 is None or d53 is None:
            return result, 0, json.dumps(td)

        b52, a52 = _touch(d52)
        b53, a53 = _touch(d53)
        if None in (b52, a52, b53, a53):
            return result, 0, json.dumps(td)
        s52, s53 = a52 - b52, a53 - b53
        joint = s52 <= TH and s53 <= TH

        pu, p52, p53 = int(pos.get(UNDER, 0)), int(pos.get(V5200, 0)), int(pos.get(V5300, 0))
        if joint:
            result[UNDER] = _mm_orders(UNDER, und, pu, MAX_SPREAD_UNDER)
            result[V5200] = _mm_orders(V5200, d52, p52, MAX_SPREAD_VEV)
            result[V5300] = _mm_orders(V5300, d53, p53, MAX_SPREAD_VEV)
        else:
            result[UNDER] = _flatten_long(UNDER, und, pu) + _flatten_short(UNDER, und, pu)
            result[V5200] = _flatten_long(V5200, d52, p52) + _flatten_short(V5200, d52, p52)
            result[V5300] = _flatten_long(V5300, d53, p53) + _flatten_short(V5300, d53, p53)

        td["_prev_tight"] = 1.0 if joint else 0.0
        td["_s5200"] = float(s52)
        td["_s5300"] = float(s53)
        ubb, uaa = _touch(und)
        if ubb and uaa:
            td["_s_under"] = float(uaa - ubb)
        return result, 0, json.dumps(td)
