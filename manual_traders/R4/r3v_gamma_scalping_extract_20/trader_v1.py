"""
Round 4 Phase 2 — **live** counterparty hook (requires backtester injection of
`__BT_TAPE_TRADES_JSON__` on each tick; see imc-prosperity-4-backtester test_runner).

**Hypothesis (from Phase 1 tape):** when **Mark 67** aggressively buys
**VELVETFRUIT_EXTRACT** (trade price ≥ best ask at that tick), short-horizon
extract mid tended to rise on historical tape — toy **follow** with small clip.

**Risk control:** skip if position already ≥ 80; max child 12; respect limits.
No other logic (Phase 2 will add burst fades in v2+).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV = [f"VEV_{k}" for k in STRIKES]
PRODUCTS = [HYDRO, EXTRACT] + VEV

LIMITS = {HYDRO: 200, EXTRACT: 200, **{v: 300 for v in VEV}}

MARK67 = "Mark 67"
MAX_EX_POS = 80
CLIP = 12


def _best(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if not depth:
        return None, None
    bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return bb, ba


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        conversions = 0
        store: dict[str, Any] = {}
        obs = getattr(state.observations, "plainValueObservations", None) or {}

        raw = obs.get("__BT_TAPE_TRADES_JSON__", "[]")
        try:
            tape = json.loads(raw) if isinstance(raw, str) else []
        except json.JSONDecodeError:
            tape = []

        fired = False
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
                fired = True
                break
        store["m67_follow"] = fired
        if "__BT_TAPE_DAY__" in obs:
            store["tape_day"] = int(obs["__BT_TAPE_DAY__"])
        return result, conversions, json.dumps(store)
