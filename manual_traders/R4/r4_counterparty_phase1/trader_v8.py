"""
Round 4 — **v7 hydro rescue: bid-only hydro in tight gate** (ablation).

`trader_v7` two-sided hydro MM lost **~2.1k** on HYDROGEL_PACK under worse fills while
raising total PnL. Hypothesis: passive **asks** on hydro are the bleeding leg when
duopoly lifts; **bids** still let us lean into tight-gate liquidity without offering
the ask into Mark14/22/49 flow.

When **joint tight**: quote **HYDROGEL_PACK** **buy side only** (passive bid+1/touch),
clip 3. **No hydro sells** in tight (skip sell-skip logic for hydro — irrelevant).
VELVET + VEV surface same as v7 (extract sell-skip; hydro sell-skip removed in tight).

**Wide:** trim long hydro like v7 (and other legs).
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
CLIP_H_BUY = 3
WIDE_TRIM = 2
SKIP_VELVET_SELL = frozenset({"Mark 22", "Mark 49"})


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
        if not buyer or seller not in SKIP_VELVET_SELL:
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
        h_sym = _sym(state, HY)
        skip_u = bool(u_sym and _skip_velvet_sell(state, u_sym))
        td["skip_u_sell"] = int(skip_u)
        td["hydro_tight_mode"] = "bid_only"

        tight_list = [U, HY] + SURFACE
        wide_list = [U, HY] + SURFACE

        if tight:
            for p in tight_list:
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
                    clip = CLIP_H_BUY
                else:
                    clip = CLIP_V
                buy_px, sell_px = _passive_prices(bb, ask)
                can_buy = max(0, min(clip, lim - cur))
                can_s = max(0, min(clip, lim + cur))
                if p == HY:
                    if can_buy > 0:
                        orders.setdefault(s, []).append(Order(s, int(buy_px), int(can_buy)))
                    continue
                if can_buy > 0:
                    orders.setdefault(s, []).append(Order(s, int(buy_px), int(can_buy)))
                if skip_u and p == U:
                    continue
                if can_s > 0:
                    orders.setdefault(s, []).append(Order(s, int(sell_px), -int(can_s)))
        else:
            for p in wide_list:
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
