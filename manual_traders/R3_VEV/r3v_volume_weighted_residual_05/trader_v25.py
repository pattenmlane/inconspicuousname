"""
vouchers_final_strategy: joint 5200+5300 spread <= 2, **passive** VEV flow + EMA-gated extract.

- Gate off: flatten U + all VEV (same as v23).
- Gate on, extract spread small: EMA(micro-mid) mean-reversion on U at touch.
- Gate on, VEV: **join bid** to add long 5200/5300 on dip (dev<=-EDGE); **join bid/ask** only —
  do not lift the ask (v23 bled on worse fills from aggressive option buys). Reduce size vs v23.
"""
from __future__ import annotations

import json
import math
from typing import Any

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
VOU = [f"VEV_{k}" for k in STRIKES]
LIMITS: dict[str, int] = {U: 200, **{f"VEV_{k}": 300 for k in STRIKES}}

TH = 2
MAX_VEV_SPREAD = 6
EMA_U_ALPHA = 0.08
U_EDGE = 2
U_Q = 12
VEV_PASSIVE_Q = 8

_CSV = "csv_day"
_EMA_U = "ema_U_mid"
_ANCHOR_S0 = {5250.0: 0, 5245.0: 1, 5267.5: 2}


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _bb_ba(d: OrderDepth) -> tuple[int | None, int | None]:
    if not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders), min(d.sell_orders)


def spread1(d: OrderDepth) -> int | None:
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def infer_csv_day(state: TradingState, store: dict[str, Any]) -> int:
    if _CSV in store:
        return int(store[_CSV])
    du = state.order_depths.get(U)
    if du is None:
        return 0
    bb, ba = _bb_ba(du)
    if bb is None or ba is None:
        return 0
    s0 = 0.5 * (bb + ba)
    best_d, best_e = 0, 1e9
    for a, d in _ANCHOR_S0.items():
        e = abs(s0 - a)
        if e < best_e:
            best_e, best_d = e, d
    if best_e > 5.0:
        best_d = 0
    store[_CSV] = best_d
    return best_d


def _update_ema_u(store: dict[str, Any], mid: float) -> float:
    prev = store.get(_EMA_U)
    if prev is not None and isinstance(prev, (int, float)) and math.isfinite(float(prev)):
        e = EMA_U_ALPHA * mid + (1.0 - EMA_U_ALPHA) * float(prev)
    else:
        e = mid
    store[_EMA_U] = e
    return e


def _joint_tight(state: TradingState) -> bool:
    d2 = state.order_depths.get(VEV_5200)
    d3 = state.order_depths.get(VEV_5300)
    if d2 is None or d3 is None:
        return False
    s2, s3 = spread1(d2), spread1(d3)
    if s2 is None or s3 is None:
        return False
    return s2 <= TH and s3 <= TH


def _flat_all(state: TradingState, du: OrderDepth) -> dict[str, list[Order]]:
    orders: dict[str, list[Order]] = {U: [], **{v: [] for v in VOU}}
    if du is not None:
        bb_u, ba_u = _bb_ba(du)
        if bb_u is not None and ba_u is not None:
            p = int(state.position.get(U, 0))
            if p > 0:
                orders[U] = [Order(U, ba_u, -min(p, 200))]
            elif p < 0:
                orders[U] = [Order(U, bb_u, min(-p, 200))]
    for v in VOU:
        p = int(state.position.get(v, 0))
        if p == 0:
            continue
        d = state.order_depths.get(v)
        if d is None:
            continue
        bb, ba = _bb_ba(d)
        if bb is None or ba is None:
            continue
        if p > 0:
            orders[v] = [Order(v, ba, -min(p, 300))]
        else:
            orders[v] = [Order(v, bb, min(-p, 300))]
    return orders


class Trader:
    def run(self, state: TradingState):
        store = _parse_td(state.traderData)
        _ = infer_csv_day(state, store)

        du = state.order_depths.get(U)
        if du is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        bb_u, ba_u = _bb_ba(du)
        if bb_u is None or ba_u is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        s_mid = 0.5 * (bb_u + ba_u)
        ema = _update_ema_u(store, s_mid)
        dev = s_mid - ema
        u_spr = int(ba_u - bb_u)

        gate = _joint_tight(state)
        orders: dict[str, list[Order]] = {U: [], **{v: [] for v in VOU}}

        if not gate:
            o = _flat_all(state, du)
            out = {k: v for k, v in o.items() if v}
            return out, 0, json.dumps(store, separators=(",", ":"))

        if u_spr > 6:
            o = _flat_all(state, du)
            return {k: v for k, v in o.items() if v}, 0, json.dumps(store, separators=(",", ":"))

        pos_u = int(state.position.get(U, 0))
        if dev <= -U_EDGE and pos_u < LIMITS[U] - U_Q:
            q = min(U_Q, LIMITS[U] - pos_u)
            if q > 0:
                orders[U].append(Order(U, ba_u, q))
        if dev >= U_EDGE and pos_u > 0:
            q = min(U_Q, pos_u)
            if q > 0:
                orders[U].append(Order(U, bb_u, -q))

        if dev <= -U_EDGE:
            for sym in (VEV_5200, VEV_5300):
                d = state.order_depths.get(sym)
                if d is None:
                    continue
                bb, ba = _bb_ba(d)
                if bb is None or ba is None or ba - bb > MAX_VEV_SPREAD:
                    continue
                pos = int(state.position.get(sym, 0))
                if pos < LIMITS[sym] - VEV_PASSIVE_Q:
                    q = min(VEV_PASSIVE_Q, LIMITS[sym] - pos)
                    if q > 0:
                        orders[sym].append(Order(sym, bb, q))

        for sym in (VEV_5200, VEV_5300):
            pos = int(state.position.get(sym, 0))
            if pos <= 0:
                continue
            d = state.order_depths.get(sym)
            if d is None:
                continue
            bb, ba = _bb_ba(d)
            if bb is None or ba is None:
                continue
            if dev >= U_EDGE and pos > 0:
                q = min(VEV_PASSIVE_Q, pos)
                if q > 0:
                    orders[sym].append(Order(sym, ba, -q))

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
