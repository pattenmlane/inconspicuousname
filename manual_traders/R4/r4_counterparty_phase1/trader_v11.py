"""
Round 4 — **v9 + Phase-1 adverse sell-skip on all tight-gate VEV surface legs**.

`trader_v9` only skips **VELVETFRUIT_EXTRACT** sells when tape shows aggressive buy
(price>=ask1) with passive seller Mark22/Mark49. Offline counts under joint tight
(`r4_v11_aggrbuy_m22_m49_counts_under_tight.csv`): same pattern exists on VEV_5100/5200/5300
(albeit rare: 2+2+1 prints over days 1–3). Apply the **same** skip rule per surface symbol.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
HY = "HYDROGEL_PACK"
G5200 = "VEV_5200"
G5300 = "VEV_5300"
SURFACE = ["VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
SPREAD_TH = 2
CLIP_U = 8
CLIP_V = 6
CLIP_H = 4
WIDE_TRIM = 2
SKIP_VELVET_SELL = frozenset({"Mark 22", "Mark 49"})
SKIP_HYDRO_SELL = frozenset({"Mark 14", "Mark 22", "Mark 49"})
SKIP_HYDRO_BUY_PASSIVE = frozenset({"Mark 22"})


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
    if product in (U, HY):
        return 200
    return 300


def _skip_sell_aggr_buy(state: TradingState, listing_sym: str, passive_sellers: frozenset) -> bool:
    mt = getattr(state, "market_trades", None) or {}
    if not isinstance(mt, dict):
        return False
    lst = mt.get(listing_sym)
    if not lst:
        return False
    depths = getattr(state, "order_depths", None) or {}
    ba = _best_bid_ask(depths.get(listing_sym))
    if ba is None:
        return False
    _, ask1 = ba
    for tr in lst:
        buyer = getattr(tr, "buyer", None) or ""
        seller = getattr(tr, "seller", None) or ""
        if not buyer or seller not in passive_sellers:
            continue
        try:
            px = int(getattr(tr, "price", 0) or 0)
        except (TypeError, ValueError):
            continue
        if px >= ask1:
            return True
    return False


def _skip_buy_aggr_sell(state: TradingState, listing_sym: str, passive_buyers: frozenset) -> bool:
    mt = getattr(state, "market_trades", None) or {}
    if not isinstance(mt, dict):
        return False
    lst = mt.get(listing_sym)
    if not lst:
        return False
    depths = getattr(state, "order_depths", None) or {}
    ba = _best_bid_ask(depths.get(listing_sym))
    if ba is None:
        return False
    bid1, _ = ba
    for tr in lst:
        buyer = getattr(tr, "buyer", None) or ""
        seller = getattr(tr, "seller", None) or ""
        if not seller or buyer not in passive_buyers:
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
        h_sym = _sym(state, HY)
        skip_u = bool(u_sym and _skip_sell_aggr_buy(state, u_sym, SKIP_VELVET_SELL))
        skip_h_sell = bool(h_sym and _skip_sell_aggr_buy(state, h_sym, SKIP_HYDRO_SELL))
        skip_h_buy = bool(
            tight and h_sym and _skip_buy_aggr_sell(state, h_sym, SKIP_HYDRO_BUY_PASSIVE)
        )
        td["skip_u_sell"] = int(skip_u)
        td["skip_h_sell"] = int(skip_h_sell)
        td["skip_h_buy_m22"] = int(skip_h_buy)

        tight_products = [U, HY] + SURFACE
        wide_products = [U, HY] + SURFACE

        if tight:
            for p in tight_products:
                s = _sym(state, p)
                if s is None:
                    continue
                ba = _best_bid_ask(depths.get(s))
                if ba is None:
                    continue
                bb, ask = ba
                lim = _lim(p)
                cur = int(pos.get(s, 0))
                if p == U:
                    clip = CLIP_U
                elif p == HY:
                    clip = CLIP_H
                else:
                    clip = CLIP_V
                buy_px, sell_px = _passive_prices(bb, ask)
                can_buy = max(0, min(clip, lim - cur))
                can_s = max(0, min(clip, lim + cur))
                if not (p == HY and skip_h_buy) and can_buy > 0:
                    orders.setdefault(s, []).append(Order(s, int(buy_px), int(can_buy)))
                skip_vev_sell = p in SURFACE and _skip_sell_aggr_buy(state, s, SKIP_VELVET_SELL)
                skip_sell = (p == U and skip_u) or skip_vev_sell or (p == HY and skip_h_sell)
                if skip_sell:
                    continue
                if can_s > 0:
                    orders.setdefault(s, []).append(Order(s, int(sell_px), -int(can_s)))
        else:
            for p in wide_products:
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
