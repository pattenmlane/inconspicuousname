"""
Round 4 Phase 3 — **Sonic joint gate** stacked on Phase 2 burst-follow.

Same burst trigger as `trader_v1.py`, but we only arm TTL / only place extract bids when
**VEV_5200** and **VEV_5300** both have BBO spread **<= 2** (ask−bid), matching
vouchers_final_strategy / R3 analysis convention.

This tests the Phase 3 thesis: counterparty burst edge **interacts** with tight voucher
surface (cleaner hedge / less noise per Sonic + inclineGod spread state).
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
G5200 = "VEV_5200"
G5300 = "VEV_5300"
W_BURST = 500
CLIP = 6
TTL_TICKS = 8
BURST_MIN_SYM = 3
SPREAD_TH = 2


def _load_burst_lists() -> dict[str, list[int]]:
    p = Path(__file__).resolve().parent / "burst_near_timestamps_by_day.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


_BURST_TS: dict[str, list[int]] | None = None


def _near_burst(day: int, ts: int) -> bool:
    global _BURST_TS
    if _BURST_TS is None:
        _BURST_TS = _load_burst_lists()
    arr = _BURST_TS.get(str(day), [])
    if not arr:
        return False
    i = bisect.bisect_left(arr, ts - W_BURST)
    while i < len(arr) and arr[i] <= ts + W_BURST:
        if abs(arr[i] - ts) <= W_BURST:
            return True
        i += 1
    return False


def _ephemeral_burst(state: TradingState) -> bool:
    mt = getattr(state, "market_trades", None) or {}
    if not isinstance(mt, dict) or not mt:
        return False
    syms: set[str] = set()
    for lst in mt.values():
        for tr in lst:
            if getattr(tr, "buyer", None) != "Mark 01" or getattr(tr, "seller", None) != "Mark 22":
                return False
            s = getattr(tr, "symbol", None)
            if s:
                syms.add(str(s))
    return len(syms) >= BURST_MIN_SYM


def _joint_tight(state: TradingState) -> bool:
    depths = getattr(state, "order_depths", None) or {}
    listings = getattr(state, "listings", {}) or {}

    def sym_for(prod: str) -> str | None:
        for sym, lst in listings.items():
            if getattr(lst, "product", None) == prod:
                return sym
        return None

    s52, s53 = sym_for(G5200), sym_for(G5300)
    if not s52 or not s53:
        return False
    for s in (s52, s53):
        d = depths.get(s)
        if d is None:
            return False
        buys = getattr(d, "buy_orders", None) or {}
        sells = getattr(d, "sell_orders", None) or {}
        if not buys or not sells:
            return False
        bb, ba = max(buys), min(sells)
        if bb >= ba:
            return False
        if ba - bb > SPREAD_TH:
            return False
    return True


def _sym(state: TradingState, product: str) -> str | None:
    listings = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _best_bid_ask(depth: OrderDepth | None):
    if depth is None:
        return None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys)
    ba = min(sells)
    if bb >= ba:
        return None
    return int(bb), int(ba)


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        day = int(getattr(state, "day_num", -1) or -1)
        ts = int(getattr(state, "timestamp", 0) or 0)
        burst = _near_burst(day, ts) or _ephemeral_burst(state)
        tight = _joint_tight(state)

        rem = int(td.get("burst_ttl", 0) or 0)
        if burst and tight:
            rem = TTL_TICKS
        elif rem > 0:
            rem -= 1
        td["burst_ttl"] = rem
        td["burst_gate"] = int(burst)
        td["tight2"] = int(tight)

        sym = _sym(state, U)
        if sym is None or rem <= 0 or not tight:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        depths = getattr(state, "order_depths", None) or {}
        pos = getattr(state, "position", None) or {}
        ba = _best_bid_ask(depths.get(sym))
        if ba is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        bb, ask = ba
        cur = int(pos.get(sym, 0))
        lim = 200
        can = max(0, min(CLIP, lim - cur))
        if can <= 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        px = bb + 1 if ask > bb + 1 else bb
        return {sym: [Order(sym, int(px), int(can))]}, 0, json.dumps(td, separators=(",", ":"))
