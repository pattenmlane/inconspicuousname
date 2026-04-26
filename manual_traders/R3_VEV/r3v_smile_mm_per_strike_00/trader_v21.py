"""
Round 3 — **vouchers_final_strategy** only (no legacy smile-MM thesis).

Sonic / `STRATEGY.txt`: trade when **VEV_5200** and **VEV_5300** both have BBO spread
**≤ TH** (ticks). **Wide regime:** flat (no quotes) — risk filter per "do not trust
small mispricings" when the surface is not tight.

**Tight regime:** simple mid-touch market making on all ten VEVs + VELVETFRUIT_EXTRACT
(mid ± half-spread), inventory skew on vouchers, no IV/Greeks. No HYDROGEL_PACK.

inclineGod: we only use per-contract **spreads** for the joint gate, not price correlation.

This abandons the prior per-strike smile / quadratic fit line of work.

Parent concept: `round3work/vouchers_final_strategy/STRATEGY.txt` + `ORIGINAL_DISCORD_QUOTES.txt`.
"""
from __future__ import annotations

import inspect
import json
import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDERLYING = "VELVETFRUIT_EXTRACT"
GATE_5200 = "VEV_5200"
GATE_5300 = "VEV_5300"
TIGHT_SPREAD_TH = 2

VEV_PRODUCTS = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
LIMITS = {**{p: 300 for p in VEV_PRODUCTS}, UNDERLYING: 200}

# Mid-touch MM (no smile)
BASE_VEV_HALF = 1.35
# Slightly wider in log-m for deep wings (execution risk) without a full IV model
WING_KM_SQ = 280.0
EXTRACT_HALF = 2.4
SIZE_VEV = 32
SIZE_EXTRACT = 16
SKEW_PER_UNIT = 0.04


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


def strike_from_product(p: str) -> float:
    return float(p.split("_", 1)[1])


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

        if UNDERLYING not in depths:
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

        for p in VEV_PRODUCTS:
            if p not in depths:
                continue
            b = book_mid(depths.get(p))
            if b is None:
                continue
            _, _, mid = b
            K = strike_from_product(p)
            km = math.log(K / max(mid_u, 1e-9))
            lim = LIMITS[p]
            pos = int(positions.get(p, 0))
            skew = SKEW_PER_UNIT * (pos / max(lim, 1))
            spr = b[1] - b[0]
            fair = mid - skew * spr
            half = BASE_VEV_HALF + WING_KM_SQ * (km**2)
            bid_p = int(round(fair - half))
            ask_p = int(round(fair + half))
            bid_p = min(bid_p, int(b[1]) - 1)
            ask_p = max(ask_p, int(b[0]) + 1)
            if bid_p >= ask_p:
                continue
            qb = min(SIZE_VEV, lim - pos)
            qs = min(SIZE_VEV, lim + pos)
            ol: list[Order] = []
            if qb > 0 and bid_p > 0:
                ol.append(Order(p, bid_p, qb))
            if qs > 0 and ask_p > 0:
                ol.append(Order(p, ask_p, -qs))
            if ol:
                orders[p] = ol

        pos_u = int(positions.get(UNDERLYING, 0))
        lim_u = LIMITS[UNDERLYING]
        bu2 = book_mid(depths.get(UNDERLYING))
        if bu2 is not None:
            fair_x = bu2[2]
            half_x = EXTRACT_HALF
            bid_x = int(round(fair_x - half_x))
            ask_x = int(round(fair_x + half_x))
            bid_x = min(bid_x, int(bu2[1]) - 1)
            ask_x = max(ask_x, int(bu2[0]) + 1)
            if bid_x < ask_x:
                qb = min(SIZE_EXTRACT, lim_u - pos_u)
                qs = min(SIZE_EXTRACT, lim_u + pos_u)
                ou: list[Order] = []
                if qb > 0:
                    ou.append(Order(UNDERLYING, bid_x, qb))
                if qs > 0:
                    ou.append(Order(UNDERLYING, ask_x, -qs))
                if ou:
                    orders[UNDERLYING] = ou

        return orders, 0, json.dumps(store, separators=(",", ":"))
