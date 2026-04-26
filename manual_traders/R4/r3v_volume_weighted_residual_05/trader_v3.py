"""
Round 4 — tape-inspired **Sonic joint gate** execution test on **VELVETFRUIT_EXTRACT** only.

Phase 3 showed pooled **K=20** forward extract mid is higher when **VEV_5200** and **VEV_5300**
top-of-book spreads are both **<= 2** (inner-join style on same timestamp as R3 script).

This trader (no hydrogel, no VEV orders):
- Computes `tight` from current `order_depths` for 5200, 5300, and U.
- **Rising edge** `tight`: buy **U** at best ask up to `TARGET_LONG` (capped by limit).
- **Falling edge** `not tight`: flatten **U** at touch (sell long at bid / buy back short).
- While `tight` unchanged: no repeat entries (hold).

`traderData` persists `_prev_tight` across ticks (resets each sim day with empty string at start;
day boundaries are handled because first tick sets prev from stored JSON).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
K5200 = "VEV_5200"
K5300 = "VEV_5300"

LIMIT_U = 200
TH = 2
TARGET_LONG = 14
MAX_U_SPREAD = 8

_KEY = "_sonic_prev_tight"


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _bb_ba(d: OrderDepth | None) -> tuple[int | None, int | None]:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders), min(d.sell_orders)


def _spr(d: OrderDepth | None) -> int | None:
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def _joint_tight(depths: dict[str, OrderDepth]) -> bool:
    s2 = _spr(depths.get(K5200))
    s3 = _spr(depths.get(K5300))
    if s2 is None or s3 is None:
        return False
    return s2 <= TH and s3 <= TH


class Trader:
    def run(self, state: TradingState):
        store = _parse_td(state.traderData)
        depths = state.order_depths

        du = depths.get(U)
        bb_u, ba_u = _bb_ba(du)
        if bb_u is None or ba_u is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        if int(ba_u - bb_u) > MAX_U_SPREAD:
            tight = False
        else:
            tight = _joint_tight(depths)

        prev = store.get(_KEY)
        prev_b = bool(prev) if isinstance(prev, bool) else None

        orders: dict[str, list[Order]] = {U: []}
        pos = int(state.position.get(U, 0))

        if prev_b is None:
            store[_KEY] = tight
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if tight and not prev_b:
            q = min(TARGET_LONG, LIMIT_U - pos)
            if q > 0:
                orders[U].append(Order(U, ba_u, q))
        elif (not tight) and prev_b:
            if pos > 0:
                orders[U].append(Order(U, bb_u, -pos))
            elif pos < 0:
                orders[U].append(Order(U, ba_u, -pos))

        store[_KEY] = tight
        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
