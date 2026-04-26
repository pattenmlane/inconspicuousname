"""
v28: v24 with +1 tick wider VEV edge in tight and wide (adverse fill protection under --worse).
vouchers_final_strategy only (STRATEGY.txt, ORIGINAL_DISCORD_QUOTES.txt): no smile/RV-IV legacy.

- Joint gate: (spread_5200 <= 2) and (spread_5300 <= 2) with spread = ask1 - bid1.
- Tight regime: smaller MM edge, larger size on U + all VEVs (Sonic: hedge into tight surface).
- Wide regime: larger edge, smaller size (inclineGod: wide book = execution risk).

Fair for each VEV: intrinsic + time_value, with time_value = max(0, mid - intrinsic) scaled by
exp(-0.5*T) so deep OTM shrinks toward intrinsic as T->0 (simple DTE proxy from round3description).
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

# Historical open extract mids -> csv_day for DTE
ANCHOR = {
    5250.0: 0,
    5245.0: 1,
    5267.5: 2,
    10011.0: 3,
}

# Tight (joint gate on)
MM_EDGE_U_T = 1
SIZE_U_T = 22
VEV_EDGE_T = 2
VEV_SIZE_T = 22

# Wide
MM_EDGE_U_W = 3
SIZE_U_W = 8
VEV_EDGE_W = 4
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


def _intrinsic(s: float, k: int) -> float:
    return max(0.0, s - float(k))


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
        s5 = _spr(d5) if d5 else 999
        s3 = _spr(d3) if d3 else 999
        tight = s5 <= TH_SPREAD and s3 <= TH_SPREAD
        td["tight_5200_5300"] = bool(tight)

        if tight:
            eu, su, ev, sav = MM_EDGE_U_T, SIZE_U_T, VEV_EDGE_T, VEV_SIZE_T
        else:
            eu, su, ev, sav = MM_EDGE_U_W, SIZE_U_W, VEV_EDGE_W, VEV_SIZE_W

        def mm(
            sym: str,
            d: OrderDepth,
            fair: float,
            edge: int,
            size: int,
        ) -> list[Order]:
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
        orders[U] = mm(U, d_u, m_u, eu, su)

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
            orders[sym] = mm(sym, d, fair, ev, sav)

        return orders, 0, json.dumps(td, separators=(",", ":"))
