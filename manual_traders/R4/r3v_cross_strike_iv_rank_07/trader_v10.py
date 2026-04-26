"""
Round 4 iteration 10 — Phase 2 cross-instrument: reverse hydro duopoly × Sonic gate → long extract.

Offline (r4_p2c_hydro_m14_m38_summary.json): Mark 14 buys HYDROGEL_PACK from Mark 38 (all rows are
agg==sell_agg in Phase1 file — i.e. trade price <= L1 bid, Mark 38 aggressive seller into M14).
Under sonic_tight, mean extract dm_ex k20 ≈ +1.08 (n=126) vs loose ≈ -0.53 (n=370).

Sim: Sonic joint gate + market_trades on hydro with buyer Mark 14, seller Mark 38, price <= best bid
(M38 hitting bid to M14) -> lift ask on VELVETFRUIT_EXTRACT (clip 18, cooldown 6), same execution
pattern as trader_v3.
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
_LAST = "last_fire_v10"


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


def _hydro_m38_sell_to_m14(state: TradingState, sym_h: str, bid_h: int) -> bool:
    for tr in (getattr(state, "market_trades", None) or {}).get(sym_h, []) or []:
        if (
            getattr(tr, "buyer", None) == "Mark 14"
            and getattr(tr, "seller", None) == "Mark 38"
            and int(getattr(tr, "price", 0)) <= int(bid_h)
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

        if not _hydro_m38_sell_to_m14(state, sym_h, int(hb)):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        bucket = ts // 100
        last = td.get(_LAST)
        if isinstance(last, int) and bucket - last < COOLDOWN:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        du = depths.get(sym_ex)
        ubb, uba = _ba(du)
        if ubb is None or uba is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        pos_e = int(pos.get(sym_ex, 0))
        qb = min(CLIP, EX_LIM - pos_e)
        if qb <= 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        td[_LAST] = int(bucket)
        return {sym_ex: [Order(sym_ex, int(uba), qb)]}, 0, json.dumps(td, separators=(",", ":"))
