"""
Round 4 — Tiered Sonic gate (v11 base).

**Core joint (Sonic):** VEV_5200 and VEV_5300 BBO spreads **both <= 2** — quote
VELVETFRUIT_EXTRACT, **all** VEV strikes, HYDROGEL (same sizes as v11).

**Wing band only:** both spreads **<= 3** but **not** core (exactly one or both in {3}
or one at 3) — quote **HYDROGEL_PACK** and **outer** VEVs (4000, 4500, 6000, 6500) only;
no extract and no inner ATM VEVs (5000–5500) so the 2-tick *surface* thesis is not
diluted on the core strip.

**Else:** same loose path as v11 (soft extract SIZE_U_SOFT=14, skip when Mark55
aggressive on extract this tick).

Tape context: analyze_r4_sonic_tier_gate_shares.py — wing_only occupies the gap
between TH=2 and TH=3 joint frequency (~71% of timestamps pooled).
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

U = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
TH_CORE = 2
TH_WING = 3
STRIKES_INNER = [5000, 5100, 5200, 5300, 5400, 5500]
STRIKES_OUTER = [4000, 4500, 6000, 6500]
VOUCHERS_INNER = [f"VEV_{k}" for k in STRIKES_INNER]
VOUCHERS_OUTER = [f"VEV_{k}" for k in STRIKES_OUTER]
STRIKES_ALL = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS_ALL = [f"VEV_{k}" for k in STRIKES_ALL]
LIMITS = {U: 200, HYDRO: 200, **{v: 300 for v in VOUCHERS_ALL}}
SIZE_U = 18
SIZE_VEV = 14
SIZE_H = 10
SIZE_U_SOFT = 14


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


def _emit_u(depths: dict[str, OrderDepth], pos: dict[str, int], size: int) -> dict[str, list[Order]]:
    out: dict[str, list[Order]] = {}
    if U not in depths:
        return out
    pr = _inside_pair(depths[U])
    if not pr:
        return out
    bid_p, ask_p = pr
    p0 = int(pos.get(U, 0))
    lim = LIMITS[U]
    qb = min(size, max(0, lim - p0))
    qs = min(size, max(0, lim + p0))
    lo: list[Order] = []
    if qb > 0:
        lo.append(Order(U, bid_p, qb))
    if qs > 0:
        lo.append(Order(U, ask_p, -qs))
    if lo:
        out[U] = lo
    return out


def _emit_hydro(depths: dict[str, OrderDepth], pos: dict[str, int], size: int) -> dict[str, list[Order]]:
    out: dict[str, list[Order]] = {}
    if HYDRO not in depths:
        return out
    prh = _inside_pair(depths[HYDRO])
    if not prh:
        return out
    bid_p, ask_p = prh
    ph = int(pos.get(HYDRO, 0))
    limh = LIMITS[HYDRO]
    qh = min(size, max(0, limh - ph))
    qhs = min(size, max(0, limh + ph))
    ho: list[Order] = []
    if qh > 0:
        ho.append(Order(HYDRO, bid_p, qh))
    if qhs > 0:
        ho.append(Order(HYDRO, ask_p, -qhs))
    if ho:
        out[HYDRO] = ho
    return out


def _emit_vevs(
    depths: dict[str, OrderDepth], pos: dict[str, int], vouchers: list[str], size: int
) -> dict[str, list[Order]]:
    orders: dict[str, list[Order]] = {}
    for v in vouchers:
        if v not in depths:
            continue
        pr = _inside_pair(depths[v])
        if not pr:
            continue
        bid_p, ask_p = pr
        p0 = int(pos.get(v, 0))
        lim = LIMITS[v]
        qb = min(size, max(0, lim - p0))
        qs = min(size, max(0, lim + p0))
        lo2: list[Order] = []
        if qb > 0:
            lo2.append(Order(v, bid_p, qb))
        if qs > 0:
            lo2.append(Order(v, ask_p, -qs))
        if lo2:
            orders[v] = lo2
    return orders


def _mark55_aggressive_extract_tick(depths: dict[str, OrderDepth], mt: dict[str, list[Trade]]) -> bool:
    d = depths.get(U)
    if d is None:
        return False
    bb, ba = _book(d)
    if bb is None or ba is None:
        return False
    bid1, ask1 = int(bb), int(ba)
    for tr in mt.get(U, []) or []:
        px = int(getattr(tr, "price", 0) or 0)
        qty = int(getattr(tr, "quantity", 0) or 0)
        if qty <= 0:
            continue
        if getattr(tr, "buyer", None) == "Mark 55" and px >= ask1:
            return True
        if getattr(tr, "seller", None) == "Mark 55" and px <= bid1:
            return True
    return False


class Trader:
    def run(self, state: TradingState):
        _ = _parse_td(getattr(state, "traderData", None))
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}
        mt: dict[str, list[Trade]] = getattr(state, "market_trades", None) or {}

        s52 = _bbo_spread(depths[V5200]) if V5200 in depths else None
        s53 = _bbo_spread(depths[V5300]) if V5300 in depths else None

        orders: dict[str, list[Order]] = {}
        if s52 is None or s53 is None:
            skip_u_soft = _mark55_aggressive_extract_tick(depths, mt)
            if not skip_u_soft:
                orders.update(_emit_u(depths, pos, SIZE_U_SOFT))
            td = {"mode": "u_soft_only", "leg_missing": True, "skip_m55": bool(skip_u_soft)}
            return orders, 0, json.dumps(td, separators=(",", ":"))

        joint_core = s52 <= TH_CORE and s53 <= TH_CORE
        joint_wing = s52 <= TH_WING and s53 <= TH_WING

        if joint_core:
            orders.update(_emit_u(depths, pos, SIZE_U))
            orders.update(_emit_hydro(depths, pos, SIZE_H))
            orders.update(_emit_vevs(depths, pos, VOUCHERS_ALL, SIZE_VEV))
            td = {"mode": "core_full", "joint_core": True}
        elif joint_wing:
            orders.update(_emit_hydro(depths, pos, SIZE_H))
            orders.update(_emit_vevs(depths, pos, VOUCHERS_OUTER, SIZE_VEV))
            td = {"mode": "wing_hydro_outer_vev", "joint_core": False, "joint_wing": True}
        else:
            skip_u_soft = _mark55_aggressive_extract_tick(depths, mt)
            if not skip_u_soft:
                orders.update(_emit_u(depths, pos, SIZE_U_SOFT))
            td = {"mode": "u_soft_only", "joint_core": False, "joint_wing": False, "skip_m55": bool(skip_u_soft)}

        return orders, 0, json.dumps(td, separators=(",", ":"))
