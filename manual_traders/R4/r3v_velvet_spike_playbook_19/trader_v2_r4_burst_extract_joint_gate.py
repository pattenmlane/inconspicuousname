"""
Round 4 Phase 3 — Mark01→Mark22 VEV basket burst + Sonic joint tight gate on 5200/5300.

Requires at current tick:
  - >= BURST_MIN_VEV tape prints VEV_* with buyer Mark 01 and seller Mark 22
  - L1 spread on VEV_5200 and VEV_5300 both <= TH (same as STRATEGY / R3 script)

Then aggressive buy VELVETFRUIT_EXTRACT at best ask (small size). Phase 3 tape:
ungated burst mean fwd20_ex ~0.30; joint-tight pooled burst rows mean ~0.295 (n=1336).
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState, Trade

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
    buys = getattr(depth, "buy_orders", {}) or {}
    sells = getattr(depth, "sell_orders", {}) or {}
    if not buys or not sells:
        return None
    return int(min(sells.keys()) - max(buys.keys()))


def _burst_m01_m22_vev_count(state: TradingState) -> int:
    n = 0
    for sym, lst in (getattr(state, "market_trades", None) or {}).items():
        if not str(sym).startswith("VEV_"):
            continue
        for t in lst or []:
            if getattr(t, "buyer", None) == "Mark 01" and getattr(t, "seller", None) == "Mark 22":
                n += 1
    return n


def _joint_tight(depths: dict[str, Any]) -> bool:
    d5 = depths.get("VEV_5200")
    d3 = depths.get("VEV_5300")
    s5, s3 = _spread(d5), _spread(d3)
    if s5 is None or s3 is None:
        return False
    return s5 <= TH and s3 <= TH


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths = getattr(state, "order_depths", {}) or {}
        if EXTRACT not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        burst = _burst_m01_m22_vev_count(state)
        jt = _joint_tight(depths)
        store["burst_m01_m22_vev"] = int(burst)
        store["joint_tight"] = bool(jt)

        d = depths[EXTRACT]
        buys = getattr(d, "buy_orders", {}) or {}
        sells = getattr(d, "sell_orders", {}) or {}
        if not buys or not sells:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        pos = int(getattr(state, "position", {}).get(EXTRACT, 0) or 0)
        out: dict[str, list[Order]] = {}

        if burst >= BURST_MIN_VEV and jt:
            ask = min(sells.keys())
            q = min(LOT, LIMIT - pos, abs(int(sells.get(ask, 0))))
            if q > 0:
                out[EXTRACT] = [Order(EXTRACT, int(ask), int(q))]

        return out, 0, json.dumps(store, separators=(",", ":"))
