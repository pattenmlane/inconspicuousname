"""
Round 4 — v6 entry + hybrid exit (v10).

v9: passive bid while tight; sell **only** on tight→wide falling edge (full
size). Under worse matching the sell often never fills → stuck long (+70 vs
v6 +252).

v10: same passive bid while tight as v6. On **falling edge**, submit **full**
long at best ask (same as v9). On **every subsequent wide** tick while still
long, keep v6’s **CLIP** sells at best ask until flat.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDER = "VELVETFRUIT_EXTRACT"
GATE_A = "VEV_5200"
GATE_B = "VEV_5300"
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


class Trader:
    def run(self, state: TradingState):
        td = getattr(state, "traderData", "") or ""
        try:
            bag: dict[str, Any] = json.loads(td) if td else {}
        except json.JSONDecodeError:
            bag = {}
        if not isinstance(bag, dict):
            bag = {}

        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < WARMUP_TICKS:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}

        sym_u = sym_for(state, UNDER)
        s52 = sym_for(state, GATE_A)
        s53 = sym_for(state, GATE_B)
        if sym_u is None or s52 is None or s53 is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        du = depths.get(sym_u)
        if du is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))
        bu_u = du.buy_orders or {}
        se_u = du.sell_orders or {}
        if not bu_u or not se_u:
            return {}, 0, json.dumps(bag, separators=(",", ":"))
        bb = int(max(bu_u))
        ba = int(min(se_u))
        if ba < bb:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        now_t = joint_tight(depths.get(s52), depths.get(s53))
        prev_t = bool(bag.get("_prev_tight", False))
        bag["_prev_tight"] = now_t

        p = int(pos.get(sym_u, 0))
        falling = (not now_t) and prev_t

        if falling and p > 0:
            return {sym_u: [Order(sym_u, ba, -p)]}, 0, json.dumps(bag, separators=(",", ":"))

        if now_t and p + CLIP <= LIM:
            return {sym_u: [Order(sym_u, bb, CLIP)]}, 0, json.dumps(bag, separators=(",", ":"))

        if (not now_t) and p > 0:
            q = min(p, CLIP)
            if q > 0:
                return {sym_u: [Order(sym_u, ba, -q)]}, 0, json.dumps(bag, separators=(",", ":"))

        return {}, 0, json.dumps(bag, separators=(",", ":"))
