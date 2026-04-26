"""
Round 4 trader_v4 — same as v1 (Sonic joint gate → one-tick MM on extract + VEV_5200
+ VEV_5300 only) with larger per-side clip to test fill-cap saturation (cf. R3 v25).

HYDROGEL_PACK not traded.
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

try:
    from prosperity4bt.constants import LIMITS
except ImportError:
    LIMITS = {
        "VELVETFRUIT_EXTRACT": 200,
        **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
    }

U = "VELVETFRUIT_EXTRACT"
G5200 = "VEV_5200"
G5300 = "VEV_5300"
_JOINT_TH = 2
_WARMUP = 5
_MM = 28


def _td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def best_bid_ask(d: OrderDepth | None) -> tuple[int | None, int | None]:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders), min(d.sell_orders)


def spread(d: OrderDepth | None) -> int | None:
    b, a = best_bid_ask(d)
    if b is None or a is None:
        return None
    return int(a - b)


def joint_tight(depths: dict[str, OrderDepth]) -> bool:
    s0 = spread(depths.get(G5200))
    s1 = spread(depths.get(G5300))
    if s0 is None or s1 is None:
        return False
    return s0 <= _JOINT_TH and s1 <= _JOINT_TH


def one_tick_mm(sym: str, d: OrderDepth | None, pos: int, lim: int, sz: int) -> list[Order]:
    b, a = best_bid_ask(d)
    if b is None or a is None or a <= b:
        return []
    sp = int(a - b)
    if sp < 2:
        return []
    if sp == 2:
        m = (b + a) // 2
        bp, ap = m, m
    else:
        bp, ap = b + 1, a - 1
        if bp >= ap:
            return []
    out: list[Order] = []
    if pos < lim:
        out.append(Order(sym, bp, min(sz, lim - pos)))
    if pos > -lim:
        out.append(Order(sym, ap, -min(sz, lim + pos)))
    return out


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        store = _td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < _WARMUP:
            return result, conversions, json.dumps(store, separators=(",", ":"))

        depths = getattr(state, "order_depths", None) or {}
        pos = getattr(state, "position", None) or {}

        if not joint_tight(depths):
            return result, conversions, json.dumps(store, separators=(",", ":"))

        for sym in (U, G5200, G5300):
            lim = LIMITS.get(sym, 200 if sym == U else 300)
            oo = one_tick_mm(sym, depths.get(sym), int(pos.get(sym, 0)), lim, _MM)
            if oo:
                result[sym] = oo

        return result, conversions, json.dumps(store, separators=(",", ":"))
