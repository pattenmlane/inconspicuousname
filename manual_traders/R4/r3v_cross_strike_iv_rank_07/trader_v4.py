"""
Round 4 iteration 4 — Phase 1 edge + Phase 3 gate (short fade).

Tape (Phase 1): Mark 22 *aggressive* sells on VEV_5300 (trade price <= L1 bid) associate with
negative short-horizon self-mid on 5300 (e.g. k=5 in participant stats).

Phase 3: require Sonic joint gate (5200+5300 spread <= 2) so execution sits in the tight-surface
regime where voucher signals were shown to be less noise-dominated.

Sim: when gate on AND market_trades show Mark 22 selling VEV_5300 at/below best bid, place a
marketable limit sell at best bid (negative quantity) up to clip, respecting voucher position -300.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH = 2
CLIP = 40
VEV_LIM = 300
COOLDOWN = 6
WARMUP = 5
_LAST = "last_fire_v4"


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _sym(state: TradingState, prod: str) -> str | None:
    for s, lst in (getattr(state, "listings", {}) or {}).items():
        if getattr(lst, "product", None) == prod:
            return s
    return None


def _ba(d: OrderDepth | None) -> tuple[int | None, int | None]:
    if d is None:
        return None, None
    b, s = getattr(d, "buy_orders", None) or {}, getattr(d, "sell_orders", None) or {}
    if not b or not s:
        return None, None
    return max(b.keys()), min(s.keys())


def _sonic(depths: dict[str, OrderDepth], s520: str, s530: str) -> bool:
    b5, a5 = _ba(depths.get(s520))
    b3, a3 = _ba(depths.get(s530))
    if None in (b5, a5, b3, a3):
        return False
    return (a5 - b5) <= TH and (a3 - b3) <= TH


def _m22_aggressive_sell_vev5300(state: TradingState, sym: str, bid: int) -> bool:
    for tr in (getattr(state, "market_trades", None) or {}).get(sym, []) or []:
        if getattr(tr, "seller", None) == "Mark 22" and int(getattr(tr, "price", 0)) <= int(bid):
            return True
    return False


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        s520 = _sym(state, "VEV_5200")
        s530_core = _sym(state, "VEV_5300")
        sym_wing = _sym(state, "VEV_5300")
        if not s520 or not s530_core or not sym_wing:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _sonic(depths, s520, s530_core):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        d = depths.get(sym_wing)
        bid, ask = _ba(d)
        if bid is None or ask is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _m22_aggressive_sell_vev5300(state, sym_wing, int(bid)):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        bucket = ts // 100
        last = td.get(_LAST)
        if isinstance(last, int) and bucket - last < COOLDOWN:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        p = int(pos.get(sym_wing, 0))
        # Position in [-VEV_LIM, +VEV_LIM]; selling decreases p; max additional sell qty = p + VEV_LIM
        room = p + VEV_LIM
        q = min(CLIP, room)
        if q <= 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        td[_LAST] = int(bucket)
        return {sym_wing: [Order(sym_wing, int(bid), -q)]}, 0, json.dumps(td, separators=(",", ":"))
