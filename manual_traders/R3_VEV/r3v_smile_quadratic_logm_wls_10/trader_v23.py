"""
Iteration 23 — vouchers_final_strategy / joint tight gate ONLY
(Sonic + inclineGod: regime from spreads; no legacy smile / IV stack.)

Thesis: round3work/vouchers_final_strategy/STRATEGY.txt and
ORIGINAL_DISCORD_QUOTES.txt — trade VELVETFRUIT_EXTRACT + VEV_* only
when VEV_5200 and VEV_5300 both have top-of-book spread <= TH (2 ticks).
Off-gate: do not trust execution / post no VEV+extract orders (Sonic: wide
surface = different regime). On-gate: one-tick inside spread on extract
and all VEVs with sp>=2 (inclineGod: per-contract book state).
No HYDROGEL_PACK.
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
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
VEVS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
_JOINT_TH = 2
_WARMUP = 5
# Gate legs: quote larger on the two surface contracts Sonic names
_GATESZ = 22
_OthersZ = 14
_EXTRACT_ON = 32


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


def tob_spread(d: OrderDepth | None) -> int | None:
    b, a = best_bid_ask(d)
    if b is None or a is None:
        return None
    return int(a - b)


def joint_tight(dmap: dict[str, OrderDepth]) -> bool:
    s0 = tob_spread(dmap.get(VEV_5200))
    s1 = tob_spread(dmap.get(VEV_5300))
    if s0 is None or s1 is None:
        return False
    return s0 <= _JOINT_TH and s1 <= _JOINT_TH


def one_tick_mm(
    product: str,
    d: OrderDepth | None,
    pos: int,
    lim: int,
    per_side: int,
) -> list[Order]:
    b, a = best_bid_ask(d)
    if b is None or a is None or a <= b:
        return []
    sp = int(a - b)
    if sp < 2:
        return []
    # sp==2: single interior tick m = b+1 == a-1
    if sp == 2:
        m = (b + a) // 2
        bid_p, ask_p = m, m
    else:
        bid_p = b + 1
        ask_p = a - 1
        if bid_p >= ask_p:
            return []
    out: list[Order] = []
    if pos < lim:
        out.append(Order(product, bid_p, min(per_side, lim - pos)))
    if pos > -lim:
        out.append(Order(product, ask_p, -min(per_side, lim + pos)))
    return out


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        store = _td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < _WARMUP:
            return result, conversions, json.dumps(store, separators=(",", ":"))

        depths: dict[str, Any] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}

        if not joint_tight(depths):
            return result, conversions, json.dumps(store, separators=(",", ":"))

        uo = one_tick_mm(
            U,
            depths.get(U),
            int(pos.get(U, 0)),
            LIMITS.get(U, 200),
            _EXTRACT_ON,
        )
        if uo:
            result[U] = uo

        for sym in VEVS:
            d = depths.get(sym)
            lim = LIMITS.get(sym, 300)
            p0 = int(pos.get(sym, 0))
            sz = _GATESZ if sym in (VEV_5200, VEV_5300) else _OthersZ
            oo = one_tick_mm(sym, d, p0, lim, sz)
            if oo:
                result[sym] = oo

        return result, conversions, json.dumps(store, separators=(",", ":"))
