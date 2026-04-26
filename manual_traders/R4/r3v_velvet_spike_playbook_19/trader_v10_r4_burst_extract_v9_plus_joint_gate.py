"""
Round 4 — Phase 3 stack: **Sonic joint tight** (VEV_5200 & VEV_5300 L1 spread <= 2) **and**
v9 burst rule (>=3 Mark01→Mark22 VEV/tick, extract L1 spread <= 5, LOT=6 / LOT=3 on csv_day 3).

Tape: at burst timestamps joint_tight is true ~99.7% of rows (r4_burst_m01_m22_rows...);
v2 showed gate non-binding for unfiltered bursts — this tests whether the gate binds
once v8/v9 extract-spread filter is applied (likely still rare / same PnL).
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


def _joint_tight(depths: dict[str, Any]) -> bool:
    s5, s3 = _spread(depths.get("VEV_5200")), _spread(depths.get("VEV_5300"))
    if s5 is None or s3 is None:
        return False
    return s5 <= TH and s3 <= TH


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

        if EXTRACT not in depths or "VEV_5200" not in depths or "VEV_5300" not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        jt = _joint_tight(depths)
        store["joint_tight"] = jt
        if not jt:
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
