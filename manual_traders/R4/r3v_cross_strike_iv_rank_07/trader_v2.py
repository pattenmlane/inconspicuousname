"""
Round 4 iteration 2 — Phase 3 (Sonic gate × Mark01→Mark22 basket on wings).

Tape: r4_p3_three_way_pair_symbol_k20.csv — under sonic_tight, Mark01|Mark22 on VEV_6000/VEV_6500
with n≈316 each shows positive mean extract forward k=20 (three-way).

Sim: when joint 5200+5300 spread<=2 AND market_trades show Mark01 buying Mark22 on VEV_6000 or VEV_6500
at/above best ask, passive join-bid those strikes (CLIP each). No hydrogel.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH = 2
CLIP = 14
WARMUP = 5
WINGS = ("VEV_6000", "VEV_6500")
VEV_LIM = 300


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


def _day(td: dict[str, Any], ts: int, s: float) -> int:
    if ts != 0:
        return int(td.get("csv_day", 0))
    h = td.get("open_S_hist")
    if not isinstance(h, list):
        h = []
    c = round(float(s), 2)
    if not h or abs(float(h[-1]) - c) > 0.25:
        h.append(c)
    td["open_S_hist"] = h[:4]
    return max(0, min(len(h) - 1, 2))


def _sonic(depths: dict[str, OrderDepth], s520: str, s530: str) -> bool:
    b5, a5 = _ba(depths.get(s520))
    b3, a3 = _ba(depths.get(s530))
    if None in (b5, a5, b3, a3):
        return False
    return (a5 - b5) <= TH and (a3 - b3) <= TH


def _basket_signal(state: TradingState) -> bool:
    """Any Mark01←Mark22 print on wing symbols this tick (tape often trades at 0 on deep calls)."""
    m = getattr(state, "market_trades", None) or {}
    for prod in WINGS:
        sym = _sym(state, prod)
        if sym is None:
            continue
        for tr in m.get(sym, []) or []:
            if getattr(tr, "buyer", None) == "Mark 01" and getattr(tr, "seller", None) == "Mark 22":
                return True
    return False


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        s520 = _sym(state, "VEV_5200")
        s530 = _sym(state, "VEV_5300")
        if not s520 or not s530:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        sym_u = _sym(state, "VELVETFRUIT_EXTRACT")
        if sym_u:
            du = depths.get(sym_u)
            ubb, uba = _ba(du)
            if ubb is not None and uba is not None:
                td["csv_day"] = _day(td, ts, 0.5 * (ubb + uba))

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _sonic(depths, s520, s530):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _basket_signal(state):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        out: dict[str, list[Order]] = {}
        for prod in WINGS:
            sym = _sym(state, prod)
            if sym is None:
                continue
            d = depths.get(sym)
            bb, ba = _ba(d)
            if bb is None or ba is None:
                continue
            buy_px = int(ba) if bb + 1 >= ba else bb + 1
            p = int(pos.get(sym, 0))
            q = min(CLIP, VEV_LIM - p)
            if q > 0:
                out.setdefault(sym, []).append(Order(sym, buy_px, q))
        return out, 0, json.dumps(td, separators=(",", ":"))
