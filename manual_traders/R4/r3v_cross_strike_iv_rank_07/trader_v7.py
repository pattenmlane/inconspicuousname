"""
Round 4 iteration 7 — Phase 1 Mark 55 on extract (aggressive sell) × Sonic gate.

Offline (r4_p1_mark55_extract_summary.json): Mark 55 aggressive *sells* on VELVETFRUIT_EXTRACT
(seller==M55, trade price <= bid) show pooled mean dm_self k=100 ≈ -0.92 (n=595, t≈-2.5);
k=5/k=20 means near zero — edge is long-horizon / slow decay not short scalp.

Sim: when Sonic joint gate AND market_trades show Mark 55 aggressively selling extract to any buyer
(price <= best bid), join the offer with a limit sell at best bid (clip 25, cooldown 5).
Tests whether the k=100 bearish signature survives worse-fill shorting under gate.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH = 2
CLIP = 25
EX_LIM = 200
COOLDOWN = 5
WARMUP = 5
_LAST = "last_fire_v7"


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


def _m55_aggressive_sell_extract(state: TradingState, sym_ex: str, bid: int) -> bool:
    for tr in (getattr(state, "market_trades", None) or {}).get(sym_ex, []) or []:
        if getattr(tr, "seller", None) == "Mark 55" and int(getattr(tr, "price", 0)) <= int(bid):
            return True
    return False


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        sym_ex = _sym(state, "VELVETFRUIT_EXTRACT")
        s520 = _sym(state, "VEV_5200")
        s530 = _sym(state, "VEV_5300")
        if not sym_ex or not s520 or not s530:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _sonic(depths, s520, s530):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        d = depths.get(sym_ex)
        bid, ask = _ba(d)
        if bid is None or ask is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _m55_aggressive_sell_extract(state, sym_ex, int(bid)):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        bucket = ts // 100
        last = td.get(_LAST)
        if isinstance(last, int) and bucket - last < COOLDOWN:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        p = int(pos.get(sym_ex, 0))
        room = p + EX_LIM
        q = min(CLIP, room)
        if q <= 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        td[_LAST] = int(bucket)
        return {sym_ex: [Order(sym_ex, int(bid), -q)]}, 0, json.dumps(td, separators=(",", ":"))
