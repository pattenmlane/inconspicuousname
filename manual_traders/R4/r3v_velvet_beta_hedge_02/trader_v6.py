"""
Round 4 — v5 with slightly larger clip sizes (grid probe).

Parent v5. Mark01→22 basket under joint gate: r4_m01_m22_fwd_extract_vev_vs_other_gate —
1336 VEV-leg rows when tight, mean extract fwd20 ~+0.30 (tape); sim size-up test.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
LIMITS = {U: 200, HYDRO: 200, **{v: 300 for v in VOUCHERS}}
TH = 2
SIZE_U = 20
SIZE_VEV = 16
SIZE_H = 12


def _bbo_spread(depth: OrderDepth) -> int | None:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if ba <= bb:
        return None
    return int(ba - bb)


def _book(depth: OrderDepth) -> tuple[int | None, int | None]:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None, None
    return max(buys.keys()), min(sells.keys())


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _inside_pair(depth: OrderDepth) -> tuple[int, int] | None:
    bb, ba = _book(depth)
    if bb is None or ba is None or ba <= bb + 1:
        return None
    bid_p, ask_p = int(bb) + 1, int(ba) - 1
    if bid_p >= ask_p:
        return None
    return bid_p, ask_p


class Trader:
    def run(self, state: TradingState):
        _ = _parse_td(getattr(state, "traderData", None))
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}

        s52 = _bbo_spread(depths[V5200]) if V5200 in depths else None
        s53 = _bbo_spread(depths[V5300]) if V5300 in depths else None
        joint = s52 is not None and s53 is not None and s52 <= TH and s53 <= TH

        orders: dict[str, list[Order]] = {}
        if not joint:
            return orders, 0, json.dumps({"joint": False}, separators=(",", ":"))

        if U in depths:
            pr = _inside_pair(depths[U])
            if pr:
                bid_p, ask_p = pr
                p0 = int(pos.get(U, 0))
                lim = LIMITS[U]
                qb = min(SIZE_U, max(0, lim - p0))
                qs = min(SIZE_U, max(0, lim + p0))
                lo: list[Order] = []
                if qb > 0:
                    lo.append(Order(U, bid_p, qb))
                if qs > 0:
                    lo.append(Order(U, ask_p, -qs))
                if lo:
                    orders[U] = lo

        if HYDRO in depths:
            prh = _inside_pair(depths[HYDRO])
            if prh:
                bid_p, ask_p = prh
                ph = int(pos.get(HYDRO, 0))
                limh = LIMITS[HYDRO]
                qh = min(SIZE_H, max(0, limh - ph))
                qhs = min(SIZE_H, max(0, limh + ph))
                ho: list[Order] = []
                if qh > 0:
                    ho.append(Order(HYDRO, bid_p, qh))
                if qhs > 0:
                    ho.append(Order(HYDRO, ask_p, -qhs))
                if ho:
                    orders[HYDRO] = ho

        for v in VOUCHERS:
            if v not in depths:
                continue
            pr = _inside_pair(depths[v])
            if not pr:
                continue
            bid_p, ask_p = pr
            p0 = int(pos.get(v, 0))
            lim = LIMITS[v]
            qb = min(SIZE_VEV, max(0, lim - p0))
            qs = min(SIZE_VEV, max(0, lim + p0))
            lo2: list[Order] = []
            if qb > 0:
                lo2.append(Order(v, bid_p, qb))
            if qs > 0:
                lo2.append(Order(v, ask_p, -qs))
            if lo2:
                orders[v] = lo2

        return orders, 0, json.dumps({"joint": True}, separators=(",", ":"))
