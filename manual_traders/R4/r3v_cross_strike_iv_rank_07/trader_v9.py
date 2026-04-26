"""
Round 4 iteration 9 — Phase 2 cross-instrument: hydro duopoly × Sonic gate → extract fade.

Offline (r4_p2c_hydro_m38_m14_summary.json): when Mark 38 buys HYDROGEL_PACK from Mark 14, pooled
extract forward dm_ex k=20 is mildly negative (~-0.21 on n≈507; |t| modest). Used as a probe
for cross-product pressure into extract under the tight voucher surface (Sonic gate).

Sim: if Sonic joint gate (5200+5300 spread<=2) AND market_trades on HYDROGEL_PACK show Mark 38
aggressively buying from Mark 14 (trade price >= best ask on hydro), short VELVETFRUIT_EXTRACT
at best bid (clip 18, cooldown 6). Respects extract short limit -200.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH = 2
CLIP = 18
EX_LIM = 200
COOLDOWN = 6
WARMUP = 5
_LAST = "last_fire_v9"


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


def _hydro_signal(state: TradingState, sym_h: str, ask_h: int) -> bool:
    for tr in (getattr(state, "market_trades", None) or {}).get(sym_h, []) or []:
        if (
            getattr(tr, "buyer", None) == "Mark 38"
            and getattr(tr, "seller", None) == "Mark 14"
            and int(getattr(tr, "price", 0)) >= int(ask_h)
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
        sym_h = _sym(state, "HYDROGEL_PACK")
        s520 = _sym(state, "VEV_5200")
        s530 = _sym(state, "VEV_5300")
        if not sym_ex or not sym_h or not s520 or not s530:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _sonic(depths, s520, s530):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        dh = depths.get(sym_h)
        hb, ha = _ba(dh)
        if hb is None or ha is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _hydro_signal(state, sym_h, int(ha)):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        bucket = ts // 100
        last = td.get(_LAST)
        if isinstance(last, int) and bucket - last < COOLDOWN:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        d = depths.get(sym_ex)
        bid, ask = _ba(d)
        if bid is None or ask is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        p = int(pos.get(sym_ex, 0))
        room = p + EX_LIM
        q = min(CLIP, room)
        if q <= 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        td[_LAST] = int(bucket)
        return {sym_ex: [Order(sym_ex, int(bid), -q)]}, 0, json.dumps(td, separators=(",", ":"))
