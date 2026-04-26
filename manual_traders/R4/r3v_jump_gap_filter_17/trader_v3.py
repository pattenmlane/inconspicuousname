"""
Round 4 iteration 3 — Phase 3 surface (v2 fix): joint gate + **crossing** quotes on VEV_5200/5300.

v2 used maker bids that could sit **at** the touch (`min(best_bid+1, floor(fair-1))` == best_bid when
fair is near mid), so the backtester never lifted from `sell_orders` and VEV PnL stayed 0.

v3: **one tick inside** the spread when possible (`best_bid+1` bid, `best_ask-1` ask), clamped by
fair±make_edge; **tighter take** (edge 1) so lift/hit triggers more often. Same extract block as v2.
"""
from __future__ import annotations

import json
import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
GATE_LEGS = (S5200, S5300)
TIGHT_TOB = 2
PRODUCTS = [
    HYDRO,
    EXTRACT,
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
LIMITS = {
    HYDRO: 200,
    EXTRACT: 200,
    **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
}
_TD = "r4v3"
_EMA = 0.15
TAKE_EDGE = 1.0
MAKE_EDGE = 1.0
MM_SZ = 16


def wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv, av = depth.buy_orders[bb], -depth.sell_orders[ba]
    tot = bv + av
    if tot <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * av + float(ba) * bv) / tot


def tob_spread(depth: OrderDepth) -> int | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return int(min(depth.sell_orders)) - int(max(depth.buy_orders))


def joint_tight_gate(state: TradingState) -> bool:
    d52 = state.order_depths.get(S5200)
    d53 = state.order_depths.get(S5300)
    if d52 is None or d53 is None:
        return False
    if not d52.buy_orders or not d52.sell_orders or not d53.buy_orders or not d53.sell_orders:
        return False
    a, b = tob_spread(d52), tob_spread(d53)
    if a is None or b is None:
        return False
    return a <= TIGHT_TOB and b <= TIGHT_TOB


def microprice(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv = float(depth.buy_orders[bb])
    av = float(abs(depth.sell_orders[ba]))
    tot = bv + av
    if tot <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * av + float(ba) * bv) / tot


def _vev_orders(
    sym: str,
    depth: OrderDepth,
    pos: int,
    lim: int,
    fair: float,
) -> list[Order]:
    """Same shape as R3 trader_v19._vev_orders: touch takers + bid/ask improved by 1 tick from touch."""
    olist: list[Order] = []
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    best_bid, best_ask = int(bb), int(ba)
    if best_ask <= fair - TAKE_EDGE + 1e-9 and pos < lim:
        q = min(MM_SZ + 10, lim - pos)
        if q > 0:
            olist.append(Order(sym, best_ask, q))
    if best_bid >= fair + TAKE_EDGE - 1e-9 and pos > -lim:
        q = min(MM_SZ + 10, lim + pos)
        if q > 0:
            olist.append(Order(sym, best_bid, -q))
    bid_anchor = int(math.floor(fair - MAKE_EDGE))
    ask_anchor = int(math.ceil(fair + MAKE_EDGE))
    bid_p = min(best_bid + 1, bid_anchor)
    bid_p = max(0, bid_p)
    if bid_p < best_ask and pos < lim:
        q = min(MM_SZ, lim - pos)
        if q > 0:
            olist.append(Order(sym, bid_p, q))
    ask_p = max(best_ask - 1, ask_anchor)
    if ask_p > best_bid and pos > -lim:
        q = min(MM_SZ, lim + pos)
        if q > 0:
            olist.append(Order(sym, ask_p, -q))
    return olist


class Trader:
    def run(self, state: TradingState):
        bu: dict[str, Any] = {}
        if state.traderData:
            try:
                o = json.loads(state.traderData)
                if isinstance(o, dict) and _TD in o and isinstance(o[_TD], dict):
                    bu = o[_TD]
            except (json.JSONDecodeError, TypeError, KeyError):
                bu = {}

        out: dict[str, list[Order]] = {p: [] for p in PRODUCTS}

        exd = state.order_depths.get(EXTRACT)
        if exd is None or not exd.buy_orders or not exd.sell_orders:
            return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))

        wm = wall_mid(exd)
        if wm is None or wm <= 0:
            return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))

        f = bu.get("fex")
        if f is None:
            f = float(wm)
        else:
            f = float(f) + _EMA * (float(wm) - float(f))
        bu["fex"] = f

        joint = joint_tight_gate(state)
        mp = microprice(exd)
        skew = 0
        if mp is not None and mp > float(wm) + 0.25:
            skew = 1
        elif mp is not None and mp < float(wm) - 0.25:
            skew = -1

        pos = int(state.position.get(EXTRACT, 0) or 0)
        lim = LIMITS[EXTRACT]
        mq = 14 if joint else 5
        edge = 2
        fi = int(round(float(f))) + skew
        bb, ba = max(exd.buy_orders), min(exd.sell_orders)
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < lim:
            out[EXTRACT].append(Order(EXTRACT, bid_p, min(mq, lim - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -lim:
            out[EXTRACT].append(Order(EXTRACT, ask_p, -min(mq, lim + pos)))

        if joint:
            for sym in GATE_LEGS:
                d = state.order_depths.get(sym)
                if d is None or not d.buy_orders or not d.sell_orders:
                    continue
                fair = wall_mid(d)
                if fair is None:
                    continue
                posv = int(state.position.get(sym, 0) or 0)
                limv = LIMITS[sym]
                out[sym].extend(_vev_orders(sym, d, posv, limv, float(fair)))

        return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))
