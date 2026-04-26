"""
Round 3 — joint tight gate, gate legs only (v37).

Parent: v34 (+19,335 worse on 4 days). Same thesis; **scale** clip size and slightly
tighter EDGE to test profit vs churn (still vouchers_final_strategy only).
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

GATE_AND_TRADE: tuple[str, str] = ("VEV_5200", "VEV_5300")
TH = 2
EMA_WINDOW = 100
EDGE = 2.5
WARMUP = 8
Q_VEV = 25
LIM_V = 300


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


def merge_orders(orders: dict[str, list[Order]]) -> dict[str, list[Order]]:
    out: dict[str, list[Order]] = {}
    for sym, lst in orders.items():
        acc: dict[int, int] = {}
        for o in lst:
            acc[o.price] = acc.get(o.price, 0) + o.quantity
        merged = [Order(sym, p, q) for p, q in sorted(acc.items()) if q != 0]
        if merged:
            out[sym] = merged
    return out


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
        last_ts = int(bag.get("_last_ts", -1))
        day = int(bag.get("_csv_day", 0))
        if last_ts >= 0 and ts < last_ts:
            day += 1
        bag["_last_ts"] = ts
        bag["_csv_day"] = day

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}

        sym52 = sym_for(state, GATE_AND_TRADE[0])
        sym53 = sym_for(state, GATE_AND_TRADE[1])
        if sym52 is None or sym53 is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))
        s52 = spread1(depths.get(sym52))
        s53 = spread1(depths.get(sym53))
        if s52 is None or s53 is None:
            return {}, 0, json.dumps(bag, separators=(",", ":"))
        if not (s52 <= TH and s53 <= TH):
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        orders: dict[str, list[Order]] = {}
        for prod in GATE_AND_TRADE:
            sym = sym_for(state, prod)
            if sym is None:
                continue
            d = depths.get(sym)
            bb, ba, mid = bbo(d)
            if mid is None or bb is None or ba is None:
                continue
            p0 = int(pos.get(sym, 0))
            em = ema_bag(bag, prod, EMA_WINDOW, float(mid))
            if mid < em - EDGE and p0 < LIM_V:
                q = min(Q_VEV, LIM_V - p0)
                if q > 0:
                    orders.setdefault(sym, []).append(Order(sym, int(ba), q))
            elif mid > em + EDGE and p0 > -LIM_V:
                q = min(Q_VEV, LIM_V + p0)
                if q > 0:
                    orders.setdefault(sym, []).append(Order(sym, int(bb), -q))

        return merge_orders(orders), 0, json.dumps(bag, separators=(",", ":"))
