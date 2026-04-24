"""
Pepper: aggressive **long 80** like ``pepper_80_0_long_only_standalone``,

plus **trim**: if touch **mid** jumps up by at least ``SPIKE_MIN`` vs previous tick’s mid,
sell up to ``TRIM_QTY`` at the **best bid** (aggressive). **Buy-back** uses the usual lift-at-ask
logic on **later ticks only** (no buy in the same tick as the trim — same-tick round-trip was
washing PnL vs pure 80/0 in the backtester).

``traderData``: ``{"prev_mid": <float|null>}``. Mids outside ``[MID_LO, MID_HI]`` are ignored
for spike detection (avoids mid=0 artifacts).
"""

from __future__ import annotations

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

PEPPER = "INTARIAN_PEPPER_ROOT"

PEPPER_TARGET_LONG = 80
PEPPER_POSITION_LIMIT = 80

SPIKE_MIN = 8.0
TRIM_QTY = 10

MID_LO = 5000.0
MID_HI = 25_000.0

# Diagnostics: incremented when ``spike`` is True (``compare_spike_trim_vs_80.py`` reads per day).
SPIKE_SIGNALS = 0


def _store_float(x: object) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _aggressive_buy_toward_80(depth: OrderDepth, position: int) -> List[Order]:
    if not depth.buy_orders or not depth.sell_orders:
        return []
    lim = max(1, PEPPER_POSITION_LIMIT)
    tgt = min(PEPPER_TARGET_LONG, lim)
    need = tgt - position
    if need <= 0:
        return []
    best_ask = min(depth.sell_orders.keys())
    ask_vol = abs(int(depth.sell_orders[best_ask]))
    q = min(need, lim - position, ask_vol)
    if q <= 0:
        return []
    return [Order(PEPPER, int(best_ask), int(q))]


def _aggressive_sell_at_bid(depth: OrderDepth, position: int, max_qty: int) -> List[Order]:
    if not depth.buy_orders or position <= 0 or max_qty <= 0:
        return []
    best_bid = max(depth.buy_orders.keys())
    bid_vol = int(depth.buy_orders[best_bid])
    q = min(max_qty, position, bid_vol)
    if q <= 0:
        return []
    return [Order(PEPPER, int(best_bid), -int(q))]


class Trader:
    def run(self, state: TradingState):
        global SPIKE_SIGNALS
        result: dict[str, list[Order]] = {}
        conversions = 0

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        depth = state.order_depths.get(PEPPER)
        if depth is None or PEPPER not in state.order_depths:
            return result, conversions, json.dumps(store)

        pos = int(state.position.get(PEPPER, 0))
        mid = _micro_mid(depth)
        prev_mid = _store_float(store.get("prev_mid"))

        orders: List[Order] = []

        spike = False
        if (
            mid is not None
            and prev_mid is not None
            and MID_LO <= mid <= MID_HI
            and MID_LO <= prev_mid <= MID_HI
            and (mid - prev_mid) >= SPIKE_MIN
        ):
            spike = True
            SPIKE_SIGNALS += 1

        trim_orders: List[Order] = []
        if spike:
            trim_orders = _aggressive_sell_at_bid(depth, pos, TRIM_QTY)
        orders.extend(trim_orders)

        trim_vol = sum(-o.quantity for o in trim_orders if o.quantity < 0)
        pos_eff = pos - trim_vol
        if not (spike and trim_vol > 0):
            orders.extend(_aggressive_buy_toward_80(depth, pos_eff))

        if orders:
            result[PEPPER] = orders

        if mid is not None and math.isfinite(mid):
            store["prev_mid"] = float(mid)
        else:
            store["prev_mid"] = store.get("prev_mid")

        return result, conversions, json.dumps(store)
