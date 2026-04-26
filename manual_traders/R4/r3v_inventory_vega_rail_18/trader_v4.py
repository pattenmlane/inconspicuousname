"""
Round 4 — Sonic joint gate, edge-triggered extract (v4).

Falsified v3: buying every tight tick under --match-trades worse is catastrophic
(STRATEGY mid vs bid/ask). v4 only **enters** once per **tight episode** (rising
edge of joint gate) with a small clip; **exits** on falling edge (flatten long
at bid). Caps entries per csv day in traderData.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDER = "VELVETFRUIT_EXTRACT"
GATE_A = "VEV_5200"
GATE_B = "VEV_5300"
TH = 2
CLIP = 8
LIM = 200
MAX_ENTRIES_PER_DAY = 4
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
        last_ts = int(bag.get("_last_ts", -1))
        day = int(bag.get("_csv_day", 1))
        if last_ts >= 0 and ts < last_ts:
            day += 1
            bag["_entries"] = 0
        bag["_last_ts"] = ts
        bag["_csv_day"] = day

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

        ent = int(bag.get("_entries", 0))
        p = int(pos.get(sym_u, 0))
        orders: dict[str, list[Order]] = {}

        if now_t and not prev_t:
            if ent < MAX_ENTRIES_PER_DAY and p + CLIP <= LIM:
                orders.setdefault(sym_u, []).append(Order(sym_u, ba, CLIP))
                bag["_entries"] = ent + 1
        elif (not now_t) and prev_t and p > 0:
            orders.setdefault(sym_u, []).append(Order(sym_u, bb, -p))

        return orders, 0, json.dumps(bag, separators=(",", ":"))
