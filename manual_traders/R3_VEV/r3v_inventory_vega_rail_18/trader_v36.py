"""
Round 3 — joint tight gate, **VELVETFRUIT_EXTRACT only** (v36).

vouchers_final_strategy: require VEV_5200 and VEV_5300 BBO spreads both <= 2.
No VEV orders (isolate extract execution vs forward-mid analysis). In-tight rows
show positive K=20 forward extract on tape; we use **small** momentum vs short EMA
with a **wide** deviation bar to reduce churn under --match-trades worse.

No HYDROGEL_PACK.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDER = "VELVETFRUIT_EXTRACT"
GATE = ("VEV_5200", "VEV_5300")
TH = 2
EMA_U = 90
EXTRACT_MOM_THR = 1.25
Q_EXTRACT = 8
LIM_U = 200
WARMUP = 10


def sym_for(state: TradingState, product: str) -> str | None:
    for s, lst in (state.listings or {}).items():
        if getattr(lst, "product", None) == product:
            return s
    return None


def bbo(depth: OrderDepth | None) -> tuple[int | None, int | None, float | None]:
    if depth is None:
        return None, None, None
    bu = depth.buy_orders or {}
    se = depth.sell_orders or {}
    if not bu or not se:
        return None, None, None
    bb = int(max(bu))
    ba = int(min(se))
    if ba < bb:
        return None, None, None
    return bb, ba, 0.5 * (bb + ba)


def spread1(depth: OrderDepth | None) -> int | None:
    bb, ba, _ = bbo(depth)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def ema_bag(bag: dict[str, Any], key: str, w: int, v: float) -> float:
    k = f"e_{key}"
    old = float(bag.get(k, v))
    a = 2.0 / (w + 1.0)
    new = a * v + (1.0 - a) * old
    bag[k] = new
    return new


class Trader:
    def run(self, state: TradingState):
        td = getattr(state, "traderData", "") or ""
        try:
            bag: dict[str, Any] = json.loads(td) if td else {}
        except json.JSONDecodeError:
            bag = {}
        if not isinstance(bag, dict):
            bag = {}

        ts = int(getattr(state, "timestamp", 0))
        last = int(bag.get("_last_ts", -1))
        day = int(bag.get("_csv_day", 0))
        if last >= 0 and ts < last:
            day += 1
        bag["_last_ts"] = ts
        bag["_csv_day"] = day

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}

        sy52 = sym_for(state, GATE[0])
        sy53 = sym_for(state, GATE[1])
        if sy52 is None or sy53 is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))
        s1 = spread1(depths.get(sy52))
        s2 = spread1(depths.get(sy53))
        if s1 is None or s2 is None or s1 > TH or s2 > TH:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        sy_u = sym_for(state, UNDER)
        if sy_u is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))
        d = depths.get(sy_u)
        bb, ba, mid = bbo(d)
        if mid is None or bb is None or ba is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        p0 = int(pos.get(sy_u, 0))
        em = ema_bag(bag, "u", EMA_U, float(mid))
        dev = float(mid) - em
        orders: dict[str, list[Order]] = {}
        if dev > EXTRACT_MOM_THR and p0 < LIM_U:
            q = min(Q_EXTRACT, LIM_U - p0)
            if q > 0:
                orders[sy_u] = [Order(sy_u, int(ba), q)]
        elif dev < -EXTRACT_MOM_THR and p0 > -LIM_U:
            q = min(Q_EXTRACT, LIM_U + p0)
            if q > 0:
                orders[sy_u] = [Order(sy_u, int(bb), -q)]

        return orders, 0, json.dumps(bag, separators=(",", ":"))
