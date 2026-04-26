"""
Round 4 — v6 + Phase 1/2 **risk-off** on Mark 01→Mark 22 VEV_5300 tape (v13).

Phase 2 / worst-pairs: aggressive Mark 01 buying VEV_5300 from Mark 22 skews
negative short-horizon on 5300 — use as **liquidity avoid** for unrelated
extract passive bids (same-tick only; reads `market_trades` from backtester).

While joint Sonic gate is tight: post passive extract bids like v6 **unless**
this tick’s tape includes buyer Mark 01, seller Mark 22 on **VEV_5300**.
Wide gate: unchanged CLIP sell at ask if long.
"""
from __future__ import annotations

from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

UNDER = "VELVETFRUIT_EXTRACT"
GATE_A = "VEV_5200"
GATE_B = "VEV_5300"
RISK_PRODUCT = "VEV_5300"
TH = 2
CLIP = 10
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
    bb = int(max(bu))
    ba = int(min(se))
    if ba < bb:
        return None
    return int(ba - bb)


def joint_tight(d52: OrderDepth | None, d53: OrderDepth | None) -> bool:
    s1 = spread_bbo(d52)
    s2 = spread_bbo(d53)
    if s1 is None or s2 is None:
        return False
    return s1 <= TH and s2 <= TH


def risk_off_01_22_5300(sym_5300: str, market_trades: dict[str, list[Trade]] | None) -> bool:
    if not market_trades:
        return False
    for tr in market_trades.get(sym_5300, []) or []:
        if getattr(tr, "buyer", None) == "Mark 01" and getattr(tr, "seller", None) == "Mark 22":
            return True
    return False


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
        sym_5300 = sym_for(state, RISK_PRODUCT)
        if sym_u is None or s52 is None or s53 is None or sym_5300 is None:
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
        mt = getattr(state, "market_trades", None)

        if tight:
            if risk_off_01_22_5300(sym_5300, mt):
                return {}, 0, getattr(state, "traderData", "") or ""
            if p + CLIP <= LIM:
                return {sym_u: [Order(sym_u, bb, CLIP)]}, 0, getattr(state, "traderData", "") or ""
            return {}, 0, getattr(state, "traderData", "") or ""

        if p > 0:
            q = min(p, CLIP)
            if q > 0:
                return {sym_u: [Order(sym_u, ba, -q)]}, 0, getattr(state, "traderData", "") or ""
        return {}, 0, getattr(state, "traderData", "") or ""
