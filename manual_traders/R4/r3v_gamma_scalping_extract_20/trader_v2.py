"""
Round 4 Phase 3 — **Sonic joint gate** + Phase 2 counterparty hook.

Only act when **VEV_5200** and **VEV_5300** both have L1 spread **≤ 2** (same
convention as round3work/vouchers_final_strategy STRATEGY / analyze script).

When gate is **on** and the tape shows **Mark 67** aggressive buy on
**VELVETFRUIT_EXTRACT** (price ≥ best ask), lift a small clip at the ask (same
as v1 but **gated**). When gate is **off**, post no orders.

Requires `__BT_TAPE_TRADES_JSON__` from the patched backtester.
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV = [f"VEV_{k}" for k in STRIKES]
PRODUCTS = [HYDRO, EXTRACT] + VEV

LIMITS = {HYDRO: 200, EXTRACT: 200, **{v: 300 for v in VEV}}

SPREAD_TH = 2
MARK67 = "Mark 67"
MAX_EX_POS = 80
CLIP = 12


def _best(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if not depth:
        return None, None
    bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return bb, ba


def _bbo_spread(depth: OrderDepth | None) -> int | None:
    bb, ba = _best(depth)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def joint_tight_gate(state: TradingState) -> bool:
    s0 = _bbo_spread(state.order_depths.get(VEV_5200))
    s1 = _bbo_spread(state.order_depths.get(VEV_5300))
    if s0 is None or s1 is None:
        return False
    return s0 <= SPREAD_TH and s1 <= SPREAD_TH


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        conversions = 0
        store: dict[str, Any] = {}
        obs = getattr(state.observations, "plainValueObservations", None) or {}

        if not joint_tight_gate(state):
            store["sonic_tight"] = False
            if "__BT_TAPE_DAY__" in obs:
                store["tape_day"] = int(obs["__BT_TAPE_DAY__"])
            return result, conversions, json.dumps(store)

        store["sonic_tight"] = True

        raw = obs.get("__BT_TAPE_TRADES_JSON__", "[]")
        try:
            tape = json.loads(raw) if isinstance(raw, str) else []
        except json.JSONDecodeError:
            tape = []

        for t in tape:
            if t.get("buyer") != MARK67 or t.get("symbol") != EXTRACT:
                continue
            pr = int(t.get("price", -1))
            d_ex = state.order_depths.get(EXTRACT)
            bb, ba = _best(d_ex)
            if bb is None or ba is None:
                continue
            if pr < ba:
                continue
            pos = int(state.position.get(EXTRACT, 0))
            if pos >= MAX_EX_POS:
                continue
            room = LIMITS[EXTRACT] - pos
            q = min(CLIP, room)
            if q > 0:
                result[EXTRACT].append(Order(EXTRACT, int(ba), int(q)))
                store["m67_follow"] = True
                break
        if "__BT_TAPE_DAY__" in obs:
            store["tape_day"] = int(obs["__BT_TAPE_DAY__"])
        return result, conversions, json.dumps(store)
