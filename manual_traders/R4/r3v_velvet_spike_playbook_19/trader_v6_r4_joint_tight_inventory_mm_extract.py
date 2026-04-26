"""
Round 4: joint-tight passive extract MM (v4) + inventory-aware skew.

v4 was +820 under --match-trades worse but -764 under --match-trades all on days 1–3,
suggesting fill-mode sensitivity from symmetric two-sided posting. This variant keeps
the same Sonic gate (VEV_5200 & VEV_5300 L1 spread <= 2) and inside-one-tick quotes,
but skews price and size toward flattening inventory: when long, lower bid / more
aggressive ask and smaller buys / larger sells; mirrored when short.
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
LOT_BASE = 8
LOT_MIN = 3
LOT_MAX = 14
# Start inventory skew above this |position| (extract lots).
POS_SOFT = 25
POS_HARD = 90
# Max extra ticks to move bid down (long) or up (short) from v4's bb+1 / ba-1.
MAX_TICK_SKEW = 2


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


def _inv_weight(pos: int) -> float:
    """0 at flat, approaches 1 as |pos| -> POS_HARD."""
    a = abs(int(pos))
    if a <= POS_SOFT:
        return 0.0
    span = max(1, POS_HARD - POS_SOFT)
    return min(1.0, float(a - POS_SOFT) / float(span))


def _skew_ticks(pos: int) -> int:
    """Signed: positive when long -> pull bid down, ask down (flatten long)."""
    w = _inv_weight(pos)
    if w <= 0.0:
        return 0
    mag = int(round(w * MAX_TICK_SKEW))
    mag = max(0, min(MAX_TICK_SKEW, mag))
    return mag if pos > 0 else (-mag if pos < 0 else 0)


def _lots(pos: int) -> tuple[int, int]:
    """(buy_lot, sell_lot) capped by position room."""
    w = _inv_weight(pos)
    if pos > 0:
        buy_lot = int(round(LOT_BASE - w * (LOT_BASE - LOT_MIN)))
        sell_lot = int(round(LOT_BASE + w * (LOT_MAX - LOT_BASE)))
    elif pos < 0:
        buy_lot = int(round(LOT_BASE + w * (LOT_MAX - LOT_BASE)))
        sell_lot = int(round(LOT_BASE - w * (LOT_BASE - LOT_MIN)))
    else:
        buy_lot = sell_lot = LOT_BASE
    buy_lot = max(LOT_MIN, min(LOT_MAX, buy_lot))
    sell_lot = max(LOT_MIN, min(LOT_MAX, sell_lot))
    return buy_lot, sell_lot


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
        sk = _skew_ticks(pos)
        store["inv_skew_ticks"] = sk

        # v4 baseline: buy at bb+1 (or bb if spread 1), sell at ba-1 (or ba); may meet at one tick.
        if ba > bb + 1:
            base_buy, base_sell = bb + 1, ba - 1
        else:
            base_buy, base_sell = bb, ba
        buy_px = int(base_buy - sk)
        sell_px = int(base_sell - sk)
        buy_px = max(bb, min(buy_px, ba))
        sell_px = min(ba, max(sell_px, bb))
        if buy_px >= sell_px and ba > bb + 1:
            buy_px, sell_px = base_buy, base_sell

        lot_b, lot_s = _lots(pos)
        store["lot_buy"] = lot_b
        store["lot_sell"] = lot_s

        o: list[Order] = []
        qb = min(lot_b, LIMIT - pos)
        if qb > 0:
            o.append(Order(EXTRACT, int(buy_px), int(qb)))
        qs = min(lot_s, LIMIT + pos)
        if qs > 0:
            o.append(Order(EXTRACT, int(sell_px), -int(qs)))
        return ({EXTRACT: o} if o else {}), 0, json.dumps(store, separators=(",", ":"))
