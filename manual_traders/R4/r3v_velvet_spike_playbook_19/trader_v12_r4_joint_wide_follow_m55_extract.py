"""
Round 4: **Not** Sonic joint tight (at least one of VEV_5200 / VEV_5300 L1 spread > 2)
AND tape Mark 55 aggressive buy on VELVETFRUIT_EXTRACT → mirror small buy at ask.

Tape: r4_mark55_extract_joint_gate_study — days 2–3 mean fwd20 higher in **wide** than
tight for M55 aggressive buys. Ablation vs v11 (tight-only).
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


def _joint_wide(depths: dict[str, Any]) -> bool:
    """True iff both voucher books exist and NOT (s5200<=TH and s5300<=TH)."""
    s5, s3 = _spread(depths.get("VEV_5200")), _spread(depths.get("VEV_5300"))
    if s5 is None or s3 is None:
        return False
    return not (s5 <= TH and s3 <= TH)


def _m55_aggr_buy_extract(state: TradingState, depths: dict[str, Any]) -> bool:
    d = depths.get(EXTRACT)
    if d is None:
        return False
    sells = getattr(d, "sell_orders", {}) or {}
    if not sells:
        return False
    ask = int(min(sells.keys()))
    for t in (getattr(state, "market_trades", {}) or {}).get(EXTRACT, []) or []:
        if getattr(t, "buyer", None) == "Mark 55" and int(getattr(t, "price", 0)) >= ask:
            return True
    return False


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths = getattr(state, "order_depths", {}) or {}
        if EXTRACT not in depths or "VEV_5200" not in depths or "VEV_5300" not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        jw = _joint_wide(depths)
        store["joint_wide"] = jw
        if not jw:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if not _m55_aggr_buy_extract(state, depths):
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
