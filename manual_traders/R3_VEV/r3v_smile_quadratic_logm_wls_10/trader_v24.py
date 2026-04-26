"""
Iteration 24 — vouchers_final_strategy: joint 5200+5300 spread gate (Sonic) +
inclineGod per-contract BBO, plus optional STRATEGY (2) layer: EMA(K=20) of
VELVETFRUIT_EXTRACT mid (same K as forward horizon in
analyze_vev_5200_5300_tight_gate_r3.py). When both gate legs are tight, lean
extract with taker at touch if |S-EMA| >= MOM_THR, else one-tick MM. All VEVs
get one-tick MM when tight (no legacy smile). No HYDROGEL_PACK.
"""
from __future__ import annotations

import json
import math
from typing import Any

from datamodel import Order, OrderDepth, TradingState

try:
    from prosperity4bt.constants import LIMITS
except ImportError:
    LIMITS = {
        "VELVETFRUIT_EXTRACT": 200,
        **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
    }

U = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
VEVS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
_JOINT_TH = 2
_WARMUP = 5
# Match analysis script: K=20 for forward; use same as EMA half-life proxy
_EMA_K = 20.0
_MOM_THR = 1.0
_TAKE_EXTRACT = 50
_GATESZ = 24
_OthersZ = 16
_EXTRACT_MM = 40


def _td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def best_bid_ask(d: OrderDepth | None) -> tuple[int | None, int | None]:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders), min(d.sell_orders)


def tob_spread(d: OrderDepth | None) -> int | None:
    b, a = best_bid_ask(d)
    if b is None or a is None:
        return None
    return int(a - b)


def mid_price(d: OrderDepth | None) -> float | None:
    b, a = best_bid_ask(d)
    if b is None or a is None:
        return None
    return 0.5 * (float(b) + float(a))


def joint_tight(dmap: dict[str, Any]) -> bool:
    s0 = tob_spread(dmap.get(VEV_5200))
    s1 = tob_spread(dmap.get(VEV_5300))
    if s0 is None or s1 is None:
        return False
    return s0 <= _JOINT_TH and s1 <= _JOINT_TH


def one_tick_mm(
    product: str,
    d: OrderDepth | None,
    pos: int,
    lim: int,
    per_side: int,
) -> list[Order]:
    b, a = best_bid_ask(d)
    if b is None or a is None or a <= b:
        return []
    sp = int(a - b)
    if sp < 2:
        return []
    if sp == 2:
        m = (b + a) // 2
        bid_p, ask_p = m, m
    else:
        bid_p = b + 1
        ask_p = a - 1
        if bid_p >= ask_p:
            return []
    out: list[Order] = []
    p = int(pos)
    if p < lim:
        out.append(Order(product, bid_p, min(per_side, lim - p)))
    if p > -lim:
        out.append(Order(product, ask_p, -min(per_side, lim + p)))
    return out


def ema_update(prev: float | None, m: float) -> float:
    a = 2.0 / (_EMA_K + 1.0)
    if prev is None or (isinstance(prev, float) and math.isnan(prev)):
        return m
    return a * m + (1.0 - a) * float(prev)


def extract_orders_tight(
    d: OrderDepth | None, pos: int, lim: int, ema: float, m: float
) -> list[Order]:
    b, a = best_bid_ask(d)
    if b is None or a is None or a <= b:
        return []
    p0 = int(pos)
    if m - ema >= _MOM_THR and p0 < lim:
        q = min(_TAKE_EXTRACT, lim - p0)
        if q > 0:
            return [Order(U, a, int(q))]
    if ema - m >= _MOM_THR and p0 > -lim:
        q = min(_TAKE_EXTRACT, lim + p0)
        if q > 0:
            return [Order(U, b, -int(q))]
    return one_tick_mm(U, d, p0, lim, _EXTRACT_MM)


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0
        store = _td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < _WARMUP:
            return result, conversions, json.dumps(store, separators=(",", ":"))

        depths: dict[str, Any] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}

        d_u = depths.get(U)
        m = mid_price(d_u)
        if m is not None and math.isfinite(m):
            prev_ema = store.get("ext_ema")
            prev_f: float | None
            if isinstance(prev_ema, (int, float)) and not isinstance(prev_ema, bool):
                prev_f = float(prev_ema)
            else:
                prev_f = None
            store["ext_ema"] = ema_update(prev_f, m)
        if not joint_tight(depths):
            return result, conversions, json.dumps(store, separators=(",", ":"))

        exf = store.get("ext_ema")
        ex = float(exf) if isinstance(exf, (int, float)) and not isinstance(exf, bool) else 0.0
        if m is not None and math.isfinite(m):
            uo = extract_orders_tight(d_u, int(pos.get(U, 0)), LIMITS.get(U, 200), ex, m)
            if uo:
                result[U] = uo
        for sym in VEVS:
            d = depths.get(sym)
            lim = LIMITS.get(sym, 300)
            p0 = int(pos.get(sym, 0))
            sz = _GATESZ if sym in (VEV_5200, VEV_5300) else _OthersZ
            oo = one_tick_mm(sym, d, p0, lim, sz)
            if oo:
                result[sym] = oo

        return result, conversions, json.dumps(store, separators=(",", ":"))
