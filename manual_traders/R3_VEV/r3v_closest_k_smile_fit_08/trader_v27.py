"""
vouchers_final_strategy (STRATEGY.txt / ORIGINAL_DISCORD_QUOTES.txt): joint 5200+5300 BBO spread gate
only for regime. Fair value per VEV: Black–Scholes call at **BBO-inverted implied vol** from that
contract's **mid** (execution-relevant, not a separate "smile chain" from neighbor strikes).

- Gate: s5200 = ask1-bid1 on VEV_5200, s5300 on VEV_5300, TH=2, tight = both <= TH.
- Tight: aggressive U+VEV; wide: defensive (same schedule as v24).

No HYDROGEL. No neighbor-k smile fit / RV-IV regime (legacy launch thesis) — only spread-state + IV from tape.
"""
from __future__ import annotations

import json
import math
from datamodel import Order, OrderDepth, TradingState
from typing import Any

U = "VELVETFRUIT_EXTRACT"
K5200 = 5200
K5300 = 5300
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VEV_BY_K = {k: f"VEV_{k}" for k in STRIKES}
ALL_VEV = list(VEV_BY_K.values())

LIMITS = {U: 200, **{v: 300 for v in ALL_VEV}}

TH_SPREAD = 2

ANCHOR = {
    5250.0: 0,
    5245.0: 1,
    5267.5: 2,
    10011.0: 3,
}

MM_EDGE_U_T = 1
SIZE_U_T = 22
VEV_EDGE_T = 1
VEV_SIZE_T = 22

MM_EDGE_U_W = 3
SIZE_U_W = 8
VEV_EDGE_W = 3
VEV_SIZE_W = 8

SKEW = 0.08


def _ba(d: OrderDepth) -> tuple[int, int] | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return max(d.buy_orders), min(d.sell_orders)


def _mid(d: OrderDepth) -> float | None:
    b = _ba(d)
    if b is None:
        return None
    return (b[0] + b[1]) / 2.0


def _spr(d: OrderDepth) -> int | None:
    b = _ba(d)
    if b is None:
        return None
    return b[1] - b[0]


def dte(csv_day: int, ts: int) -> float:
    return max((8.0 - float(csv_day)) - ((int(ts) // 100) / 10_000.0), 1e-6)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call(s: float, k: float, t: float, sig: float) -> float:
    if t <= 1e-12 or sig <= 1e-12:
        return max(s - k, 0.0)
    v = sig * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sig * sig * t) / v
    d2 = d1 - v
    return s * _ncdf(d1) - k * _ncdf(d2)


def implied_vol(mid_px: float, s: float, k: float, t: float) -> float | None:
    intrinsic = max(s - k, 0.0)
    if mid_px <= intrinsic + 1e-6 or mid_px >= s - 1e-6 or s <= 0 or k <= 0 or t <= 1e-12:
        return None
    lo, hi = 1e-4, 12.0
    if bs_call(s, k, t, lo) - mid_px > 0 or bs_call(s, k, t, hi) - mid_px < 0:
        return None
    for _ in range(34):
        m = 0.5 * (lo + hi)
        if bs_call(s, k, t, m) >= mid_px:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        depths = state.order_depths
        pos = state.position
        ts = int(state.timestamp)
        d_u = depths.get(U)
        if not d_u:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        m_u = _mid(d_u)
        if m_u is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        s = float(m_u)

        if ts == 0:
            key = round(m_u * 2) / 2.0
            if key in ANCHOR:
                td["csv_day"] = ANCHOR[key]
            else:
                ne = min(ANCHOR, key=lambda x: abs(x - m_u))
                if abs(ne - m_u) < 2.5:
                    td["csv_day"] = ANCHOR[ne]
        csv_day = int(td.get("csv_day", 0))
        t_y = dte(csv_day, ts) / 365.0

        d5 = depths.get(VEV_BY_K[K5200])
        d3 = depths.get(VEV_BY_K[K5300])
        sp5 = _spr(d5) if d5 else 999
        sp3 = _spr(d3) if d3 else 999
        tight = sp5 <= TH_SPREAD and sp3 <= TH_SPREAD
        td["tight_5200_5300"] = bool(tight)

        if tight:
            eu, su, ev, sav = MM_EDGE_U_T, SIZE_U_T, VEV_EDGE_T, VEV_SIZE_T
        else:
            eu, su, ev, sav = MM_EDGE_U_W, SIZE_U_W, VEV_EDGE_W, VEV_SIZE_W

        def mm(sym: str, d: OrderDepth, fair: float, edge: int, size: int) -> list[Order]:
            b = _ba(d)
            if b is None:
                return []
            bb, ap = b
            p = int(pos.get(sym, 0))
            sk = int(round(SKEW * p))
            bid = min(bb + 1, int(math.floor(fair - edge - sk)))
            ask = max(ap - 1, int(math.ceil(fair + edge - sk)))
            lim = LIMITS[sym]
            o: list[Order] = []
            if p < lim and bid > 0:
                o.append(Order(sym, bid, min(size, lim - p)))
            if p > -lim and ask > 0:
                o.append(Order(sym, ask, -min(size, lim + p)))
            return o

        orders: dict[str, list[Order]] = {}
        orders[U] = mm(U, d_u, s, eu, su)

        for k in STRIKES:
            sym = VEV_BY_K[k]
            d = depths.get(sym)
            if not d:
                continue
            mv = _mid(d)
            if mv is None:
                continue
            iv = implied_vol(float(mv), s, float(k), t_y)
            if iv is None:
                fair = float(mv)
            else:
                fair = bs_call(s, float(k), t_y, iv)
            orders[sym] = mm(sym, d, fair, ev, sav)

        return orders, 0, json.dumps(td, separators=(",", ":"))
