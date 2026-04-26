"""
Round 4 — baseline **vouchers_final_strategy** extract leg (Sonic joint gate).

Same mechanics as R3 `trader_v23`: **VEV_5200** and **VEV_5300** BBO spreads both ≤2
⇒ quote **VELVETFRUIT_EXTRACT** only (mid-style, EXTRACT_HALF=2.4). Off-gate: flat.
**No HYDROGEL_PACK**, no VEV orders — counterparty-aware strategy comes in later
iterations; this file anchors **Round 4** backtest harness + dual match modes.

Tape days: 1–3 in Prosperity4Data/ROUND_4 (backtester round 4).
"""
from __future__ import annotations

import inspect
import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDERLYING = "VELVETFRUIT_EXTRACT"
GATE_5200 = "VEV_5200"
GATE_5300 = "VEV_5300"
TIGHT_SPREAD_TH = 2

EXTRACT_HALF = 2.4
SIZE_EXTRACT = 16
SKEW_PER_UNIT = 0.04
LONG_LEAN_TICKS = 0.15

LIMIT_U = 200


def _csv_day_from_backtest_stack() -> int | None:
    for fr in inspect.stack():
        data = fr.frame.f_locals.get("data")
        if data is not None and hasattr(data, "day_num"):
            try:
                return int(getattr(data, "day_num"))
            except (TypeError, ValueError):
                continue
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def book_mid(depth: OrderDepth | None) -> tuple[float, float, float] | None:
    if depth is None:
        return None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if ba <= bb:
        return None
    return float(bb), float(ba), 0.5 * (bb + ba)


def bbo_spread_ticks(depth: OrderDepth | None) -> int | None:
    b = book_mid(depth)
    if b is None:
        return None
    return int(b[1] - b[0])


def joint_tight_gate(
    depths: dict[str, Any], th: int = TIGHT_SPREAD_TH
) -> tuple[bool, int | None, int | None]:
    s5 = bbo_spread_ticks(depths.get(GATE_5200))
    s3 = bbo_spread_ticks(depths.get(GATE_5300))
    if s5 is None or s3 is None:
        return False, s5, s3
    return (s5 <= th and s3 <= th), s5, s3


class Trader:
    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths: dict[str, Any] = getattr(state, "order_depths", None) or {}
        positions = getattr(state, "position", None) or {}

        csv_day = _csv_day_from_backtest_stack()
        if csv_day is None:
            csv_day = int(store.get("csv_day_hint", 0))
        store["csv_day_hint"] = csv_day

        if UNDERLYING not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        bu = book_mid(depths.get(UNDERLYING))
        if bu is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        tight, s5, s3 = joint_tight_gate(depths, TIGHT_SPREAD_TH)
        store["s5200_spread"] = s5
        store["s5300_spread"] = s3
        store["tight_two_leg"] = tight

        if not tight:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        pos_u = int(positions.get(UNDERLYING, 0))
        spr = bu[1] - bu[0]
        skew = SKEW_PER_UNIT * (pos_u / max(LIMIT_U, 1))
        fair = bu[2] - skew * spr
        lean = max(0.0, 1.0 - max(pos_u, 0) / max(LIMIT_U, 1)) * LONG_LEAN_TICKS
        half_b = EXTRACT_HALF - lean
        half_a = EXTRACT_HALF + lean
        bid_x = int(round(fair - half_b))
        ask_x = int(round(fair + half_a))
        bid_x = min(bid_x, int(bu[1]) - 1)
        ask_x = max(ask_x, int(bu[0]) + 1)
        if bid_x >= ask_x:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ou: list[Order] = []
        qb = min(SIZE_EXTRACT, LIMIT_U - pos_u)
        qs = min(SIZE_EXTRACT, LIMIT_U + pos_u)
        if qb > 0:
            ou.append(Order(UNDERLYING, bid_x, qb))
        if qs > 0:
            ou.append(Order(UNDERLYING, ask_x, -qs))
        if not ou:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        return {UNDERLYING: ou}, 0, json.dumps(store, separators=(",", ":"))
