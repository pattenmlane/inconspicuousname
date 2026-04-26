"""
Round 4 Phase 2 — executable test: **burst-follow extract** (Phase 1 edge #1).

When the tape is within **W=500** timestamp units of a **Mark 01 → Mark 22** multi-VEV
burst (>=3 symbols at same (day,timestamp)), go **long VELVETFRUIT_EXTRACT** passively
(bid+1 or touch) with small clip for a few ticks (TTL in traderData).

Offline timestamps: `burst_near_timestamps_by_day.json` (written by r4_phase2_analysis.py).
Runtime: `state.day_num` and `state.market_trades` (populated by backtester for Round 4).

Position limits per round4work/round4description.txt.
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
W_BURST = 500
CLIP = 6
TTL_TICKS = 8
BURST_MIN_SYM = 3


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
        gate = _near_burst(day, ts) or _ephemeral_burst(state)

        rem = int(td.get("burst_ttl", 0) or 0)
        if gate:
            rem = TTL_TICKS
        elif rem > 0:
            rem -= 1
        td["burst_ttl"] = rem
        td["burst_gate"] = int(gate)

        sym = _sym(state, U)
        if sym is None or rem <= 0:
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
        orders = {sym: [Order(sym, int(px), int(can))]}
        return orders, 0, json.dumps(td, separators=(",", ":"))
