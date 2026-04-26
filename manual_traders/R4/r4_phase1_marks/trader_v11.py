"""
Round 4 — **v11**: **v9** with **HYDROGEL_PACK** knobs from **env** (same pattern as **v10**).

**R4_HYDRO_HALF** (default ``2.2``), **R4_HYDRO_SIZE** (default ``14``). Extract + VEV legs
unchanged from **v9**.

Use for composing **v9**-style stacks with hydro sweeps without editing **v9**.
"""
from __future__ import annotations

import inspect
import json
import math
import os
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
GATE_5200 = "VEV_5200"
GATE_5300 = "VEV_5300"
QUOTE_VEVS = (GATE_5200, GATE_5300)
TIGHT_SPREAD_TH = 2

EXTRACT_HALF_BASE = 2.4
EXTRACT_HALF_MULT_TIGHT = 1.05
SIZE_EXTRACT_BASE = 16
SIZE_MULT_TIGHT = 1.0
SKEW_PER_UNIT = 0.04
LONG_LEAN_TICKS = 0.15

HYDRO_HALF_BASE = float(os.environ.get("R4_HYDRO_HALF", "2.2"))
SIZE_HYDRO_BASE = max(1, int(os.environ.get("R4_HYDRO_SIZE", "14")))

BASE_VEV_HALF = 1.35
WING_KM_SQ = 280.0
SIZE_VEV_ATM = 20

LIMIT_U = 200
LIMIT_H = 200
LIMITS = {GATE_5200: 300, GATE_5300: 300, UNDERLYING: 200}


def strike_from_product(p: str) -> float:
    return float(p.split("_", 1)[1])


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
        store["hydro_half_env"] = HYDRO_HALF_BASE
        store["hydro_size_env"] = SIZE_HYDRO_BASE

        for req in (UNDERLYING, GATE_5200, GATE_5300, HYDRO):
            if req not in depths:
                return {}, 0, json.dumps(store, separators=(",", ":"))

        bu = book_mid(depths.get(UNDERLYING))
        if bu is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        mid_u = bu[2]

        tight, s5, s3 = joint_tight_gate(depths, TIGHT_SPREAD_TH)
        store["s5200_spread"] = s5
        store["s5300_spread"] = s3
        store["tight_two_leg"] = tight

        if not tight:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        orders: dict[str, list[Order]] = {}

        sz = max(1, int(round(SIZE_EXTRACT_BASE * SIZE_MULT_TIGHT)))
        store["size_extract_eff"] = sz
        half0 = EXTRACT_HALF_BASE * EXTRACT_HALF_MULT_TIGHT
        store["extract_half_eff"] = half0

        pos_u = int(positions.get(UNDERLYING, 0))
        spr = bu[1] - bu[0]
        skew = SKEW_PER_UNIT * (pos_u / max(LIMIT_U, 1))
        fair = bu[2] - skew * spr
        lean = max(0.0, 1.0 - max(pos_u, 0) / max(LIMIT_U, 1)) * LONG_LEAN_TICKS
        half_b = half0 - lean
        half_a = half0 + lean
        bid_x = int(round(fair - half_b))
        ask_x = int(round(fair + half_a))
        bid_x = min(bid_x, int(bu[1]) - 1)
        ask_x = max(ask_x, int(bu[0]) + 1)
        if bid_x < ask_x:
            ou: list[Order] = []
            qb = min(sz, LIMIT_U - pos_u)
            qs = min(sz, LIMIT_U + pos_u)
            if qb > 0:
                ou.append(Order(UNDERLYING, bid_x, qb))
            if qs > 0:
                ou.append(Order(UNDERLYING, ask_x, -qs))
            if ou:
                orders[UNDERLYING] = ou

        bh = book_mid(depths.get(HYDRO))
        if bh is not None:
            pos_h = int(positions.get(HYDRO, 0))
            spr_h = bh[1] - bh[0]
            skew_h = SKEW_PER_UNIT * (pos_h / max(LIMIT_H, 1))
            fair_h = bh[2] - skew_h * spr_h
            half_h = HYDRO_HALF_BASE
            bid_h = int(round(fair_h - half_h))
            ask_h = int(round(fair_h + half_h))
            bid_h = min(bid_h, int(bh[1]) - 1)
            ask_h = max(ask_h, int(bh[0]) + 1)
            if bid_h < ask_h:
                sh = SIZE_HYDRO_BASE
                qhb = min(sh, LIMIT_H - pos_h)
                qhs = min(sh, LIMIT_H + pos_h)
                oh: list[Order] = []
                if qhb > 0:
                    oh.append(Order(HYDRO, bid_h, qhb))
                if qhs > 0:
                    oh.append(Order(HYDRO, ask_h, -qhs))
                if oh:
                    orders[HYDRO] = oh

        for p in QUOTE_VEVS:
            b = book_mid(depths.get(p))
            if b is None:
                continue
            _, _, mid = b
            K = strike_from_product(p)
            km = math.log(K / max(mid_u, 1e-9))
            lim = LIMITS[p]
            pos = int(positions.get(p, 0))
            skew = SKEW_PER_UNIT * (pos / max(lim, 1))
            spr2 = b[1] - b[0]
            fair_v = mid - skew * spr2
            half = BASE_VEV_HALF + WING_KM_SQ * (km**2)
            bid_p = int(round(fair_v - half))
            ask_p = int(round(fair_v + half))
            bid_p = min(bid_p, int(b[1]) - 1)
            ask_p = max(ask_p, int(b[0]) + 1)
            if bid_p >= ask_p:
                continue
            szv = SIZE_VEV_ATM
            qb = min(szv, lim - pos)
            qs = min(szv, lim + pos)
            ol: list[Order] = []
            if qb > 0 and bid_p > 0:
                ol.append(Order(p, bid_p, qb))
            if qs > 0 and ask_p > 0:
                ol.append(Order(p, ask_p, -qs))
            if ol:
                orders[p] = ol

        if not orders:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        return orders, 0, json.dumps(store, separators=(",", ":"))
