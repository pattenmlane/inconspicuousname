"""
Round 4 — v4 + **symmetric adverse skip on aggressive sells** (mirror analysis).

Tape analysis (`r4_p5_adverse_aggrsell_fwd20_by_passive_buyer.csv`): on VELVETFRUIT_EXTRACT
**aggressive sells** (price <= bid1), passive **Mark 49** as **buyer** has the **most negative**
mean fwd+20 (~-2.73, n=15). We skip **buy** quotes on extract for that tick when the same
pattern appears in `state.market_trades`.

Keeps v4's skip of **sell** quotes when aggressive buy lifts **Mark 22 / Mark 49** asks.
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
SKIP_SELL_IF_PASSIVE_SELLER = frozenset({"Mark 22", "Mark 49"})
SKIP_BUY_IF_PASSIVE_BUYER = frozenset({"Mark 49"})


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
    mt = getattr(state, "market_trades", None) or {}
    if not isinstance(mt, dict):
        return False
    lst = mt.get(u_sym) or mt.get(U)
    if not lst:
        return False
    depths = getattr(state, "order_depths", None) or {}
    ba = _best_bid_ask(depths.get(u_sym))
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


def _skip_velvet_buy(state: TradingState, u_sym: str) -> bool:
    """Aggressive sell through bid with passive buyer Mark 49."""
    mt = getattr(state, "market_trades", None) or {}
    if not isinstance(mt, dict):
        return False
    lst = mt.get(u_sym) or mt.get(U)
    if not lst:
        return False
    depths = getattr(state, "order_depths", None) or {}
    ba = _best_bid_ask(depths.get(u_sym))
    if ba is None:
        return False
    bid1, _ = ba
    for tr in lst:
        buyer = getattr(tr, "buyer", None) or ""
        seller = getattr(tr, "seller", None) or ""
        if not seller or buyer not in SKIP_BUY_IF_PASSIVE_BUYER:
            continue
        try:
            px = int(getattr(tr, "price", 0) or 0)
        except (TypeError, ValueError):
            continue
        if px <= bid1:
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
        skip_u_buy = bool(u_sym and _skip_velvet_buy(state, u_sym))
        td["skip_u_sell"] = int(skip_u_sell)
        td["skip_u_buy"] = int(skip_u_buy)

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
                if not (p == U and skip_u_buy) and can_buy > 0:
                    orders.setdefault(s, []).append(Order(s, int(buy_px), int(can_buy)))
                if not (p == U and skip_u_sell) and can_s > 0:
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
