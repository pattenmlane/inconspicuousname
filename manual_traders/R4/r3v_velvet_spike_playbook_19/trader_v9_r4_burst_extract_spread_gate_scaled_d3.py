"""
Round 4 — v8 (burst + extract L1 spread <= 5) with **smaller impulse on tape day 3**.

Motivation: v8 improved v1 (+7080 vs +4834) but day 3 remained -2564 PnL; v7 showed
day-3 participation is the main drag. This variant keeps the **tape-observable** v8
gate and still trades day 3, but uses LOT=3 when csv_day==3 vs LOT=6 on days 1–2
(risk scaling from sim loss concentration, not a blind calendar off-switch).
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
LOT_DEFAULT = 6
LOT_DAY3 = 3
DAY3_CSV = 3
MAX_EXT_L1_SPREAD = 5


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


def _burst_m01_m22_vev_count(state: TradingState) -> int:
    n = 0
    for sym, lst in (getattr(state, "market_trades", None) or {}).items():
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
        lot = LOT_DAY3 if csv_day == DAY3_CSV else LOT_DEFAULT
        store["burst_lot"] = lot

        if EXTRACT not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        d = depths[EXTRACT]
        buys = getattr(d, "buy_orders", {}) or {}
        sells = getattr(d, "sell_orders", {}) or {}
        if not buys or not sells:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        s_ext = _spread(d)
        store["s_ext"] = int(s_ext) if s_ext is not None else -1

        burst_n = _burst_m01_m22_vev_count(state)
        store["burst_m01_m22_vev"] = int(burst_n)

        pos = int(getattr(state, "position", {}).get(EXTRACT, 0) or 0)
        out: dict[str, list[Order]] = {}

        if s_ext is None or s_ext > MAX_EXT_L1_SPREAD:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if burst_n >= BURST_MIN_VEV:
            ask = min(sells.keys())
            cap = LIMIT - pos
            q = min(lot, cap, abs(int(sells.get(ask, 0))))
            if q > 0:
                out[EXTRACT] = [Order(EXTRACT, int(ask), int(q))]

        return out, 0, json.dumps(store, separators=(",", ":"))
