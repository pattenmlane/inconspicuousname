"""
Round 4 follow-up: passive two-sided quotes on VELVETFRUIT_EXTRACT only when Sonic joint
tight (5200 & 5300 L1 spread <=2). Small size; no VEV legs (tests gate as trade regime).
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
LIMIT = 200
LOT = 8


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


def _joint_tight(depths: dict[str, Any]) -> bool:
    s5, s3 = _spread(depths.get("VEV_5200")), _spread(depths.get("VEV_5300"))
    if s5 is None or s3 is None:
        return False
    return s5 <= TH and s3 <= TH


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths = getattr(state, "order_depths", {}) or {}
        if EXTRACT not in depths or "VEV_5200" not in depths or "VEV_5300" not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        jt = _joint_tight(depths)
        store["joint_tight"] = jt
        if not jt:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        d = depths[EXTRACT]
        b = getattr(d, "buy_orders", {}) or {}
        s = getattr(d, "sell_orders", {}) or {}
        if not b or not s:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        bb, ba = max(b.keys()), min(s.keys())
        pos = int(getattr(state, "position", {}).get(EXTRACT, 0) or 0)
        buy_px = bb + 1 if ba > bb + 1 else bb
        sell_px = ba - 1 if ba > bb + 1 else ba
        o: list[Order] = []
        qb = min(LOT, LIMIT - pos)
        if qb > 0:
            o.append(Order(EXTRACT, int(buy_px), int(qb)))
        qs = min(LOT, LIMIT + pos)
        if qs > 0:
            o.append(Order(EXTRACT, int(sell_px), -int(qs)))
        return ({EXTRACT: o} if o else {}), 0, json.dumps(store, separators=(",", ":"))
