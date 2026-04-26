"""
Round 4 — Sonic joint gate + extract drift bias (v3).

Tape motivation (Phase 3, r4_phase3_gate_forward_extract_k20.json): on the aligned
grid, mean 20-tick **forward extract mid** is higher when **VEV_5200** and
**VEV_5300** BBO spreads are both **<= 2** than when either is wide.

Execution (not mid): when the gate is **on**, lean **long extract** (small clip
at best ask). When the gate turns **off**, **flatten** extract toward zero at
best bid if long (or best ask if short). No HYDROGEL / no VEV orders in this
minimal test of the population gate edge under the backtester.

Position limit: 200 (constants.LIMITS).
"""
from __future__ import annotations

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDER = "VELVETFRUIT_EXTRACT"
GATE_A = "VEV_5200"
GATE_B = "VEV_5300"
TH = 2
CLIP = 12
LIM = 200
WARMUP_TICKS = 15


def sym_for(state: TradingState, product: str) -> str | None:
    for s, lst in (state.listings or {}).items():
        if getattr(lst, "product", None) == product:
            return s
    return None


def spread_bbo(depth: OrderDepth | None) -> int | None:
    if depth is None:
        return None
    bu = depth.buy_orders or {}
    se = depth.sell_orders or {}
    if not bu or not se:
        return None
    best_bid = max(bu)
    best_ask = min(se)
    if best_ask < best_bid:
        return None
    return int(best_ask - best_bid)


def joint_tight(d52: OrderDepth | None, d53: OrderDepth | None) -> bool:
    s1 = spread_bbo(d52)
    s2 = spread_bbo(d53)
    if s1 is None or s2 is None:
        return False
    return s1 <= TH and s2 <= TH


class Trader:
    def run(self, state: TradingState):
        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < WARMUP_TICKS:
            return {}, 0, getattr(state, "traderData", "") or ""

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}

        sym_u = sym_for(state, UNDER)
        s52 = sym_for(state, GATE_A)
        s53 = sym_for(state, GATE_B)
        if sym_u is None or s52 is None or s53 is None:
            return {}, 0, getattr(state, "traderData", "") or ""

        du = depths.get(sym_u)
        if du is None:
            return {}, 0, getattr(state, "traderData", "") or ""
        bu_u = du.buy_orders or {}
        se_u = du.sell_orders or {}
        if not bu_u or not se_u:
            return {}, 0, getattr(state, "traderData", "") or ""
        bb = int(max(bu_u))
        ba = int(min(se_u))
        if ba < bb:
            return {}, 0, getattr(state, "traderData", "") or ""

        tight = joint_tight(depths.get(s52), depths.get(s53))
        p = int(pos.get(sym_u, 0))

        if tight:
            if p + CLIP <= LIM:
                return {sym_u: [Order(sym_u, ba, CLIP)]}, 0, getattr(state, "traderData", "") or ""
            return {}, 0, getattr(state, "traderData", "") or ""

        # gate off: flatten long inventory (tape: weaker forward when wide)
        if p > 0:
            q = min(p, CLIP)
            if q > 0:
                return {sym_u: [Order(sym_u, bb, -q)]}, 0, getattr(state, "traderData", "") or ""
        if p < 0:
            q = min(-p, CLIP)
            if q > 0:
                return {sym_u: [Order(sym_u, ba, q)]}, 0, getattr(state, "traderData", "") or ""
        return {}, 0, getattr(state, "traderData", "") or ""
