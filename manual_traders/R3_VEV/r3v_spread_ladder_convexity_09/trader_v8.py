"""
v8: parent v7; same convexity-fade thesis, with smaller size per signal.

Grid around v7 found CLIP_QTY=6 as best/tied-best; this keeps z/window unchanged
for a simpler update.

  Z_HI=2.15, Z_LO=0.8, BF_WINDOW=400, CLIP_QTY=6.

Execution unchanged: passive one tick inside the spread when width>1.
"""
from __future__ import annotations

import json
import math
from collections import deque
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

VEV_WING_LO = "VEV_5000"
VEV_BODY = "VEV_5100"
VEV_WING_HI = "VEV_5200"
FLY_LEGS = (VEV_WING_LO, VEV_BODY, VEV_WING_HI)

Z_HI = 2.15
Z_LO = 0.8
BF_WINDOW = 400
CLIP_QTY = 6


def _sym(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _best_bid_ask(depth: OrderDepth | None) -> tuple[int, int, int, int] | None:
    if depth is None:
        return None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if bb >= ba:
        return None
    return bb, abs(int(buys[bb])), ba, abs(int(sells[ba]))


def _sell_wing_px(book: tuple[int, int, int, int]) -> int:
    bb, _, ask, _ = book
    return ask - 1 if ask > bb + 1 else bb


def _buy_body_px(book: tuple[int, int, int, int]) -> int:
    bb, _, ask, _ = book
    return bb + 1 if ask > bb + 1 else ask


def _buy_wing_px(book: tuple[int, int, int, int]) -> int:
    bb, _, ask, _ = book
    return bb + 1 if ask > bb + 1 else ask


def _sell_body_px(book: tuple[int, int, int, int]) -> int:
    bb, _, ask, _ = book
    return ask - 1 if ask > bb + 1 else bb


class Trader:
    BF_HIST = "bf_hist"

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        hist: list[float] = store.get(self.BF_HIST)
        if not isinstance(hist, list):
            hist = []
        bf_deque: deque[float] = deque(
            (float(x) for x in hist if isinstance(x, (int, float))),
            maxlen=BF_WINDOW,
        )

        syms = {p: _sym(state, p) for p in FLY_LEGS}
        if any(v is None for v in syms.values()):
            store[self.BF_HIST] = list(bf_deque)
            return {}, 0, json.dumps(store, separators=(",", ":"))

        depths: dict[str, OrderDepth | None] = getattr(state, "order_depths", None) or {}
        mids: dict[str, float] = {}
        books: dict[str, tuple[int, int, int, int]] = {}
        for p, sym in syms.items():
            book = _best_bid_ask(depths.get(sym))
            if book is None:
                store[self.BF_HIST] = list(bf_deque)
                return {}, 0, json.dumps(store, separators=(",", ":"))
            bb, _, ba, _ = book
            mids[p] = 0.5 * (float(bb) + float(ba))
            books[p] = book

        bf = mids[VEV_WING_LO] + mids[VEV_WING_HI] - 2.0 * mids[VEV_BODY]
        bf_deque.append(bf)
        store[self.BF_HIST] = list(bf_deque)

        if len(bf_deque) < BF_WINDOW:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        mu = sum(bf_deque) / BF_WINDOW
        var = sum((x - mu) ** 2 for x in bf_deque) / BF_WINDOW
        z = (bf - mu) / math.sqrt(max(var, 1e-12))

        pos = getattr(state, "position", None) or {}
        lim = 300

        def cap_buy(sym: str, q: int) -> int:
            return max(0, min(q, lim - int(pos.get(sym, 0))))

        def cap_sell(sym: str, q: int) -> int:
            return max(0, min(q, lim + int(pos.get(sym, 0))))

        prev_mode = str(store.get("fly_mode", "flat"))
        if prev_mode == "flat" and z > Z_HI:
            mode = "short"
        elif prev_mode == "short" and z < Z_LO:
            mode = "flat"
        else:
            mode = prev_mode

        store["fly_mode"] = mode
        store["last_z"] = round(float(z), 4)

        def q_short() -> int:
            return min(
                CLIP_QTY,
                cap_sell(syms[VEV_WING_LO], CLIP_QTY),
                cap_sell(syms[VEV_WING_HI], CLIP_QTY),
                cap_buy(syms[VEV_BODY], 2 * CLIP_QTY) // 2,
            )

        def q_cover() -> int:
            return min(
                CLIP_QTY,
                cap_buy(syms[VEV_WING_LO], CLIP_QTY),
                cap_buy(syms[VEV_WING_HI], CLIP_QTY),
                cap_sell(syms[VEV_BODY], 2 * CLIP_QTY) // 2,
            )

        orders: dict[str, list[Order]] = {}
        if prev_mode == "flat" and mode == "short":
            q = q_short()
            if q > 0:
                wlo, bod, whi = books[VEV_WING_LO], books[VEV_BODY], books[VEV_WING_HI]
                orders[syms[VEV_WING_LO]] = [Order(syms[VEV_WING_LO], _sell_wing_px(wlo), -q)]
                orders[syms[VEV_BODY]] = [Order(syms[VEV_BODY], _buy_body_px(bod), 2 * q)]
                orders[syms[VEV_WING_HI]] = [Order(syms[VEV_WING_HI], _sell_wing_px(whi), -q)]
        elif prev_mode == "short" and mode == "flat":
            q = q_cover()
            if q > 0:
                wlo, bod, whi = books[VEV_WING_LO], books[VEV_BODY], books[VEV_WING_HI]
                orders[syms[VEV_WING_LO]] = [Order(syms[VEV_WING_LO], _buy_wing_px(wlo), q)]
                orders[syms[VEV_BODY]] = [Order(syms[VEV_BODY], _sell_body_px(bod), -2 * q)]
                orders[syms[VEV_WING_HI]] = [Order(syms[VEV_WING_HI], _buy_wing_px(whi), q)]

        if not orders:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        return orders, 0, json.dumps(store, separators=(",", ":"))
