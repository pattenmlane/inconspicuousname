"""
Round 4 follow-up: Mark01→Mark22 VEV burst when Sonic joint gate is **wide** (NOT both <=2).

Tape has only n=3 such prints (all day 3); mean fwd20 extract at those ticks = +2 (mid).
Sim: aggressive buy at ask when pattern fires (same size as v1 for comparability).
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
BURST_MIN_VEV = 3
LOT = 6
LIMIT = 200


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def _spread(depth: OrderDepth | None) -> int | None:
    if depth is None:
        return None
    b = getattr(depth, "buy_orders", {}) or {}
    s = getattr(depth, "sell_orders", {}) or {}
    if not b or not s:
        return None
    return int(min(s.keys()) - max(b.keys()))


def _burst_m01_m22_vev(state: TradingState) -> int:
    n = 0
    for sym, lst in (getattr(state, "market_trades", None) or {}).items():
        if not str(sym).startswith("VEV_"):
            continue
        for t in lst or []:
            if getattr(t, "buyer", None) == "Mark 01" and getattr(t, "seller", None) == "Mark 22":
                n += 1
    return n


def _joint_wide(depths: dict[str, Any]) -> bool:
    s5, s3 = _spread(depths.get("VEV_5200")), _spread(depths.get("VEV_5300"))
    if s5 is None or s3 is None:
        return False
    return not (s5 <= TH and s3 <= TH)


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths = getattr(state, "order_depths", {}) or {}
        if EXTRACT not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        burst = _burst_m01_m22_vev(state)
        wide = _joint_wide(depths)
        store["burst_m01_m22_vev"] = burst
        store["joint_wide"] = wide

        d = depths[EXTRACT]
        b = getattr(d, "buy_orders", {}) or {}
        s = getattr(d, "sell_orders", {}) or {}
        if not b or not s:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        pos = int(getattr(state, "position", {}).get(EXTRACT, 0) or 0)
        if burst >= BURST_MIN_VEV and wide:
            ask = min(s.keys())
            q = min(LOT, LIMIT - pos, abs(int(s.get(ask, 0))))
            if q > 0:
                return {EXTRACT: [Order(EXTRACT, int(ask), int(q))]}, 0, json.dumps(store, separators=(",", ":"))
        return {}, 0, json.dumps(store, separators=(",", ":"))
