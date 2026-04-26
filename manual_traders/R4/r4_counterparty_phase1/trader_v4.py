"""
Round 4 — **tight-surface MM + counterparty adverse-selection skip** (Phase 1 + Phase 3).

Same Sonic joint gate and surface universe as `trader_v3.py`. Additionally, when the
tape at this timestamp shows an **aggressive buy** on **VELVETFRUIT_EXTRACT** (buyer
non-empty) with **passive seller Mark 22 or Mark 49**, we **omit sell-side quotes** on
extract only for that tick — those rows had the **highest** mean same-symbol fwd+20 in
`r4_p1_adverse_aggrbuy_fwd20_by_passive_seller.csv` (informed-style lift against passive
ask liquidity).

VEV legs unchanged vs v3 (counterparty skip is extract-only first ablation).
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
G5200 = "VEV_5200"
G5300 = "VEV_5300"
SURFACE = ["VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
SPREAD_TH = 2
CLIP_U = 8
CLIP_V = 6
WIDE_TRIM = 2
# Passive sellers on extract aggressive-buys with worst historical fwd20 (Phase 1 table)
SKIP_SELL_IF_PASSIVE_SELLER = frozenset({"Mark 22", "Mark 49"})


def _sym(state: TradingState, product: str) -> str | None:
    listings = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _best_bid_ask(depth: OrderDepth | None):
    if depth is None:
        return None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys)
    ba = min(sells)
    if bb >= ba:
        return None
    return int(bb), int(ba)


def _passive_prices(bb: int, ask: int) -> tuple[int, int]:
    if ask <= bb:
        return bb, ask
    if ask > bb + 1:
        return bb + 1, ask - 1
    return bb, ask


def _joint_tight(state: TradingState) -> bool:
    depths = getattr(state, "order_depths", None) or {}
    listings = getattr(state, "listings", {}) or {}

    def sym_for(prod: str) -> str | None:
        for sym, lst in listings.items():
            if getattr(lst, "product", None) == prod:
                return sym
        return None

    for prod in (G5200, G5300):
        s = sym_for(prod)
        if not s:
            return False
        ba = _best_bid_ask(depths.get(s))
        if ba is None:
            return False
        if ba[1] - ba[0] > SPREAD_TH:
            return False
    return True


def _lim(product: str) -> int:
    return 200 if product == U else 300


def _skip_velvet_sell(state: TradingState, u_sym: str) -> bool:
    """True if tape shows aggressive buy on VELVET vs Mark22/Mark49 (price at/through ask)."""
    mt = getattr(state, "market_trades", None) or {}
    if not isinstance(mt, dict):
        return False
    lst = mt.get(u_sym) or mt.get(U)
    if not lst:
        return False
    depths = getattr(state, "order_depths", None) or {}
    d = depths.get(u_sym)
    ba = _best_bid_ask(d)
    if ba is None:
        return False
    _, ask1 = ba
    for tr in lst:
        buyer = getattr(tr, "buyer", None) or ""
        seller = getattr(tr, "seller", None) or ""
        if not buyer or seller not in SKIP_SELL_IF_PASSIVE_SELLER:
            continue
        try:
            px = int(getattr(tr, "price", 0) or 0)
        except (TypeError, ValueError):
            continue
        if px >= ask1:
            return True
    return False


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        tight = _joint_tight(state)
        td["tight2"] = int(tight)

        depths = getattr(state, "order_depths", None) or {}
        pos = getattr(state, "position", None) or {}
        orders: dict = {}
        u_sym = _sym(state, U)
        skip_u_sell = bool(u_sym and _skip_velvet_sell(state, u_sym))
        td["skip_u_sell"] = int(skip_u_sell)

        if tight:
            for p in [U] + SURFACE:
                s = _sym(state, p)
                if s is None:
                    continue
                ba = _best_bid_ask(depths.get(s))
                if ba is None:
                    continue
                bb, ask = ba
                lim = _lim(p)
                cur = int(pos.get(s, 0))
                clip = CLIP_U if p == U else CLIP_V
                buy_px, sell_px = _passive_prices(bb, ask)
                can_buy = max(0, min(clip, lim - cur))
                can_s = max(0, min(clip, lim + cur))
                if can_buy > 0:
                    orders.setdefault(s, []).append(Order(s, int(buy_px), int(can_buy)))
                if p == U and skip_u_sell:
                    continue
                if can_s > 0:
                    orders.setdefault(s, []).append(Order(s, int(sell_px), -int(can_s)))
        else:
            for p in [U] + SURFACE:
                s = _sym(state, p)
                if s is None:
                    continue
                ba = _best_bid_ask(depths.get(s))
                if ba is None:
                    continue
                bb, ask = ba
                lim = _lim(p)
                cur = int(pos.get(s, 0))
                q = min(WIDE_TRIM, abs(cur))
                if q <= 0:
                    continue
                if cur > 0:
                    px = ask - 1 if ask > bb + 1 else ask
                    orders.setdefault(s, []).append(Order(s, int(px), -int(q)))
                else:
                    px = bb + 1 if ask > bb + 1 else bb
                    orders.setdefault(s, []).append(Order(s, int(px), int(q)))

        return orders, 0, json.dumps(td, separators=(",", ":"))
