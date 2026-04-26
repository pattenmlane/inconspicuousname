"""
Round 4 — v6 passive extract + Mark 67 **arming window** (v12).

v11 required Mark 67→22/49 extract **on the same tick** as a passive bid; that
almost never coincides with the sparse worse-fill buy (v11 got 0 PnL).

v12: when **joint tight** and `market_trades` contains Mark 67 buying extract
from Mark 22 or 49, set an arm-until timestamp (current + ARM_MS). While
**still tight** and `timestamp < arm_until`, post the same passive bids as v6.
Wide gate: same CLIP exit at ask as v6. Arm clears when gate goes wide.

Causal: uses only current-tick tape + persisted arm deadline in traderData.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

UNDER = "VELVETFRUIT_EXTRACT"
GATE_A = "VEV_5200"
GATE_B = "VEV_5300"
TH = 2
CLIP = 10
LIM = 200
WARMUP_TICKS = 15
BUYER = "Mark 67"
SELLERS = frozenset({"Mark 22", "Mark 49"})
# Worst-case spacing (v6 worse buy on day 3): Mark 67 print then ~2.32M until bid fill.
ARM_MS = 3_000_000


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


def mark67_extract_trigger(sym_u: str, market_trades: dict[str, list[Trade]] | None) -> bool:
    if not market_trades:
        return False
    for tr in market_trades.get(sym_u, []) or []:
        if getattr(tr, "buyer", None) == BUYER and getattr(tr, "seller", None) in SELLERS:
            return True
    return False


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

        tight = joint_tight(depths.get(s52), depths.get(s53))
        p = int(pos.get(sym_u, 0))
        mt = getattr(state, "market_trades", None)

        if not tight:
            out_td = json.dumps(bag, separators=(",", ":"))
            if p > 0:
                q = min(p, CLIP)
                if q > 0:
                    return {sym_u: [Order(sym_u, ba, -q)]}, 0, out_td
            return {}, 0, out_td

        if mark67_extract_trigger(sym_u, mt):
            bag["_arm_from"] = ts

        arm_from = int(bag.get("_arm_from", 0))
        if arm_from > 0 and ts - arm_from > ARM_MS:
            bag.pop("_arm_from", None)
            arm_from = 0

        allow_bid = arm_from > 0 and 0 <= ts - arm_from <= ARM_MS

        if allow_bid and p + CLIP <= LIM:
            return {sym_u: [Order(sym_u, bb, CLIP)]}, 0, json.dumps(bag, separators=(",", ":"))

        return {}, 0, json.dumps(bag, separators=(",", ":"))
