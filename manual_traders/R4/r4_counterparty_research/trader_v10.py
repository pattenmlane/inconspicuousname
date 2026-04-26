"""
Round 4 MM — v10 = v5 + larger tight-regime VEV quote size.

v5 uses VEV_SIZE_T = 22 in tight gate. v10 uses 24 (still capped by LIMITS 300).
Extract tight/loose schedule and Mark22 post-fill VEV widen unchanged from v5.
"""
from __future__ import annotations

import json
import math
from datamodel import Order, OrderDepth, TradingState
from typing import Any

U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
K5200 = 5200
K5300 = 5300
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VEV_BY_K = {k: f"VEV_{k}" for k in STRIKES}
ALL_VEV = list(VEV_BY_K.values())

LIMITS = {U: 200, H: 200, **{v: 300 for v in ALL_VEV}}

TH_SPREAD = 2
MARK_22 = "Mark 22"
SUBMISSION = "SUBMISSION"

ANCHOR = {
    5245.0: 1,
    5267.5: 2,
    5295.5: 3,
}

MM_EDGE_U_T = 1
SIZE_U_T = 22
VEV_EDGE_T = 1
VEV_SIZE_T = 24

MM_EDGE_U_W = 3
SIZE_U_W = 8
VEV_EDGE_W = 3
VEV_SIZE_W = 8

MM_EDGE_H_T = 2
SIZE_H_T = 10
MM_EDGE_H_W = 4
SIZE_H_W = 6

MM_EDGE_U_TIGHT_EXTRACT = 0
SIZE_U_TIGHT_EXTRACT = 24

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
    return max((5.0 - float(csv_day)) - ((int(ts) // 100) / 10_000.0), 1e-6)


def _intrinsic(s: float, k: int) -> float:
    return max(0.0, s - float(k))


def _vev_bought_from_m22_last_tick(own_trades: dict[str, list]) -> set[str]:
    out: set[str] = set()
    for sym, lst in own_trades.items():
        if not sym.startswith("VEV_"):
            continue
        for t in lst:
            if t.quantity > 0 and getattr(t, "buyer", None) == SUBMISSION and t.seller == MARK_22:
                out.add(sym)
    return out


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        depths = state.order_depths
        pos = state.position
        ts = int(state.timestamp)
        m22_syms = _vev_bought_from_m22_last_tick(state.own_trades or {})

        d_u = depths.get(U)
        if not d_u:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        m_u = _mid(d_u)
        if m_u is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if ts == 0:
            key = round(m_u * 2) / 2.0
            if key in ANCHOR:
                td["csv_day"] = ANCHOR[key]
            else:
                ne = min(ANCHOR, key=lambda x: abs(x - m_u))
                if abs(ne - m_u) < 2.5:
                    td["csv_day"] = ANCHOR[ne]
        csv_day = int(td.get("csv_day", 1))
        t_y = dte(csv_day, ts) / 365.0

        d5 = depths.get(VEV_BY_K[K5200])
        d3 = depths.get(VEV_BY_K[K5300])
        s5 = _spr(d5) if d5 else 999
        s3 = _spr(d3) if d3 else 999
        tight = s5 <= TH_SPREAD and s3 <= TH_SPREAD
        td["tight_5200_5300"] = bool(tight)
        td["mark22_vev_prev_tick"] = sorted(m22_syms)

        if tight:
            eu_u, su_u = MM_EDGE_U_TIGHT_EXTRACT, SIZE_U_TIGHT_EXTRACT
            ev, sav = VEV_EDGE_T, VEV_SIZE_T
            eh, sh = MM_EDGE_H_T, SIZE_H_T
        else:
            eu_u, su_u = MM_EDGE_U_W, SIZE_U_W
            ev, sav = VEV_EDGE_W, VEV_SIZE_W
            eh, sh = MM_EDGE_H_W, SIZE_H_W

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
        orders[U] = mm(U, d_u, m_u, eu_u, su_u)

        d_h = depths.get(H)
        if d_h:
            m_h = _mid(d_h)
            if m_h is not None:
                orders[H] = mm(H, d_h, m_h, eh, sh)

        for k in STRIKES:
            sym = VEV_BY_K[k]
            d = depths.get(sym)
            if not d:
                continue
            mv = _mid(d)
            if mv is None:
                continue
            intrinsic = _intrinsic(m_u, k)
            raw_tv = max(0.0, float(mv) - intrinsic)
            time_mult = math.exp(-0.5 * t_y)
            fair = intrinsic + raw_tv * time_mult
            ev_sym = ev
            sav_sym = sav
            if sym in m22_syms:
                ev_sym += 1
                sav_sym = max(4, sav_sym - 2)
            orders[sym] = mm(sym, d, fair, ev_sym, sav_sym)

        return orders, 0, json.dumps(td, separators=(",", ":"))
