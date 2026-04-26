"""
Round 4 iteration 6 — Phase 1 adverse selection (Mark 22 → Mark 49 on extract) × Sonic gate.

Tape: r4_p1_mark22_seller_markout_by_buyer_k20.csv flagged Mark 49 as worst buyer when Mark 22 is
aggressive seller (voucher tape). On VELVETFRUIT_EXTRACT the Mark22|Mark49 pair is rare (n≈12) but
sell_agg-only rows (Mark 22 hits bid; Mark 49 buys) show negative pooled forward self-mid on extract
at k=5/20/100 in offline table (see r4_p1_m22_m49_extract_markout_summary.json).

Sim: when Sonic joint gate (5200+5300 spread<=2) AND market_trades show Mark 22 selling extract to
Mark 49 at a price at/below best bid (aggressive sell), post a marketable limit sell at best bid
(negative quantity), clip 30, cooldown 8. Respects extract position limit 200 short.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH = 2
CLIP = 30
EX_LIM = 200
COOLDOWN = 8
WARMUP = 5
_LAST = "last_fire_v6"


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


def _m22_to_m49_aggressive_sell_extract(state: TradingState, sym_ex: str, bid: int) -> bool:
    for tr in (getattr(state, "market_trades", None) or {}).get(sym_ex, []) or []:
        if (
            getattr(tr, "seller", None) == "Mark 22"
            and getattr(tr, "buyer", None) == "Mark 49"
            and int(getattr(tr, "price", 0)) <= int(bid)
        ):
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

        if not _m22_to_m49_aggressive_sell_extract(state, sym_ex, int(bid)):
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
