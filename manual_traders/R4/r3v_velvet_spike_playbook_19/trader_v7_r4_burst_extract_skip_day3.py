"""
Round 4 — same as trader_v1_r4_burst_extract (Mark01→Mark22 multi-VEV burst → buy extract),
but **no trades on tape day 3** (state.csv_day == 3). Tests whether v1's large day-3
drawdown is structural to that day vs the burst signal on days 1–2.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
LIMIT = 200
BURST_MIN_VEV = 3
LOT = 6
SKIP_CSV_DAY = 3


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def _burst_m01_m22_vev_count(state: TradingState) -> int:
    mt = getattr(state, "market_trades", None) or {}
    n = 0
    for sym, lst in mt.items():
        if not str(sym).startswith("VEV_"):
            continue
        for t in lst or []:
            if getattr(t, "buyer", None) == "Mark 01" and getattr(t, "seller", None) == "Mark 22":
                n += 1
    return n


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths = getattr(state, "order_depths", {}) or {}
        csv_day = int(getattr(state, "csv_day", 0) or 0)
        store["csv_day"] = csv_day

        if EXTRACT not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        d = depths[EXTRACT]
        buys = getattr(d, "buy_orders", {}) or {}
        sells = getattr(d, "sell_orders", {}) or {}
        if not buys or not sells:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        burst_n = _burst_m01_m22_vev_count(state)
        store["burst_m01_m22_vev"] = int(burst_n)

        pos = int(getattr(state, "position", {}).get(EXTRACT, 0) or 0)
        out: dict[str, list[Order]] = {}

        if csv_day == SKIP_CSV_DAY:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if burst_n >= BURST_MIN_VEV:
            ask = min(sells.keys())
            cap = LIMIT - pos
            q = min(LOT, cap, abs(int(sells.get(ask, 0))))
            if q > 0:
                out[EXTRACT] = [Order(EXTRACT, int(ask), int(q))]

        return out, 0, json.dumps(store, separators=(",", ":"))
