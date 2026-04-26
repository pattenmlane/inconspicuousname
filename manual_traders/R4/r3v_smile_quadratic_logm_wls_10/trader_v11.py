"""
Round 4 trader_v11 — v5 with larger per-side clip on core VEV_5200 / VEV_5300 only (24 vs 18).

Phase 1 motivation: r4_p1_mark_product_cross.csv shows very high n on Mark 22 × VEV_5200 /
VEV_5300 rows (e.g. n=160 K5 on 5300) with short-horizon same-symbol forward structure; v4
raised global MM size with no PnL change, so this tests whether the basket core strikes are
clip-saturated separately from extract (v10) and wings (v8).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

try:
    from prosperity4bt.constants import LIMITS
except ImportError:
    LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
    }

U = "VELVETFRUIT_EXTRACT"
HY = "HYDROGEL_PACK"
G5200 = "VEV_5200"
G5300 = "VEV_5300"
VEV_SYMS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
_JOINT_TH = 2
_WARMUP = 5
_MM_CORE = 24
_MM_WING = 10
_MM_HY = 8
_MM_EX = 18


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


def _mm_size(sym: str) -> int:
    if sym == HY:
        return _MM_HY
    if sym == U:
        return _MM_EX
    if sym in (G5200, G5300):
        return _MM_CORE
    return _MM_WING


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

        for sym in [U, HY, *VEV_SYMS]:
            lim = LIMITS.get(sym, 200)
            oo = one_tick_mm(sym, depths.get(sym), int(pos.get(sym, 0)), lim, _mm_size(sym))
            if oo:
                result[sym] = oo

        return result, conversions, json.dumps(store, separators=(",", ":"))
