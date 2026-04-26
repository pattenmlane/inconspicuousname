"""
Round 4: if Sonic joint tight (5200 & 5300 L1 <=2) AND tape shows Mark 67 aggressive buy
on VELVETFRUIT_EXTRACT this tick (price >= best ask from book), mirror with small buy at ask.

Tape study (r4_mark67_aggr_buy_extract_fwd20_by_gate_day.csv): fwd20 mid positive in both
tight and wide; tight subset is small n per day — this sim tests execution overlap only.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
TH = 2
LOT = 4
LIMIT = 200


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def _spread(d: OrderDepth | None) -> int | None:
    if d is None:
        return None
    b, s = getattr(d, "buy_orders", {}) or {}, getattr(d, "sell_orders", {}) or {}
    if not b or not s:
        return None
    return int(min(s.keys()) - max(b.keys()))


def _joint_tight(depths: dict[str, Any]) -> bool:
    a, b = _spread(depths.get("VEV_5200")), _spread(depths.get("VEV_5300"))
    return a is not None and b is not None and a <= TH and b <= TH


def _m67_aggr_buy_extract(state: TradingState, depths: dict[str, Any]) -> bool:
    d = depths.get(EXTRACT)
    if d is None:
        return False
    sells = getattr(d, "sell_orders", {}) or {}
    if not sells:
        return False
    ask = int(min(sells.keys()))
    for t in (getattr(state, "market_trades", {}) or {}).get(EXTRACT, []) or []:
        if getattr(t, "buyer", None) == "Mark 67" and int(getattr(t, "price", 0)) >= ask:
            return True
    return False


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths = getattr(state, "order_depths", {}) or {}
        if EXTRACT not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if not _joint_tight(depths):
            store["joint_tight"] = False
            return {}, 0, json.dumps(store, separators=(",", ":"))

        store["joint_tight"] = True
        if not _m67_aggr_buy_extract(state, depths):
            return {}, 0, json.dumps(store, separators=(",", ":"))

        d = depths[EXTRACT]
        sells = getattr(d, "sell_orders", {}) or {}
        if not sells:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        ask = min(sells.keys())
        pos = int(getattr(state, "position", {}).get(EXTRACT, 0) or 0)
        q = min(LOT, LIMIT - pos, abs(int(sells.get(ask, 0))))
        if q <= 0:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        return {EXTRACT: [Order(EXTRACT, int(ask), int(q))]}, 0, json.dumps(store, separators=(",", ":"))
