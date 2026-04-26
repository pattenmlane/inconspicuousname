"""
Round 4 — **v5 + Mark 55 aggressive-sell extract hook** (Phase 1 adverse-selection signal).

- **v5**: Sonic-tight extract mid\u00b11 size 12; v3-style hydro/VEVs; Mark 38 (no ask) / Mark 67 (+bid size) on extract.
- **v6**: same as v5, plus when **market_trades** on extract show **Mark 55** **aggressively selling**
  (trade price **\u2264** bid, seller id **Mark 55**), set **extract bid size to 0** for this tick (do not lean
  into lift-offer when tape says sell pressure at touch).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEVS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
G520 = "VEV_5200"
G530 = "VEV_5300"
GATE_MAX = 2

LIMITS = {H: 200, U: 200, **{v: 300 for v in VEVS}}

HALF = 2
HALF_U_TIGHT = 1
BASE_SZ = 6
BASE_SZ_U_TIGHT = 12
BOOST_BID = 2


def best_bid_ask(d: OrderDepth) -> tuple[int | None, int | None]:
    if not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders.keys()), min(d.sell_orders.keys())


def micro_mid(d: OrderDepth) -> float | None:
    bb, ba = best_bid_ask(d)
    if bb is None or ba is None:
        return None
    return 0.5 * (bb + ba)


def spread_top(d: OrderDepth) -> int | None:
    bb, ba = best_bid_ask(d)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def sonic_tight(depths: dict) -> bool:
    d52 = depths.get(G520)
    d53 = depths.get(G530)
    if d52 is None or d53 is None:
        return False
    s52 = spread_top(d52)
    s53 = spread_top(d53)
    if s52 is None or s53 is None:
        return False
    return s52 <= GATE_MAX and s53 <= GATE_MAX


def extract_trade_flags(state: TradingState) -> tuple[bool, bool, bool]:
    """Returns (m38_aggressive_buy, m67_aggressive_buy, m55_aggressive_sell) on extract."""
    mt = getattr(state, "market_trades", None) or {}
    trades = mt.get(U)
    if not trades:
        return False, False, False
    depths = getattr(state, "order_depths", {}) or {}
    du = depths.get(U)
    if du is None:
        return False, False, False
    bb, ba = best_bid_ask(du)
    if bb is None or ba is None:
        return False, False, False
    m38 = m67 = m55_sell = False
    for t in trades:
        buyer = getattr(t, "buyer", None)
        seller = getattr(t, "seller", None)
        pr = int(getattr(t, "price", 0))
        if buyer == "Mark 38" and pr >= ba:
            m38 = True
        if buyer == "Mark 67" and pr >= ba:
            m67 = True
        if seller == "Mark 55" and pr <= bb:
            m55_sell = True
    return m38, m67, m55_sell


class Trader:
    def run(self, state: TradingState):
        store: dict[str, Any] = {}
        raw = getattr(state, "traderData", "") or ""
        if raw:
            try:
                o = json.loads(raw)
                if isinstance(o, dict):
                    store = o
            except (json.JSONDecodeError, TypeError):
                store = {}

        pos = getattr(state, "position", {}) or {}
        depths = getattr(state, "order_depths", {}) or {}
        out: dict[str, list[Order]] = {}

        tight = sonic_tight(depths)
        m38_buy, m67_buy, m55_sell = extract_trade_flags(state)

        for sym in [H, U, *VEVS]:
            d = depths.get(sym)
            if d is None:
                continue
            mid = micro_mid(d)
            if mid is None:
                continue
            bb, ba = best_bid_ask(d)
            if bb is None or ba is None:
                continue
            fair = float(mid)

            if sym in (U, H) and not tight:
                continue
            if sym in (G520, G530) and not tight:
                continue

            if sym == U and tight:
                half = int(HALF_U_TIGHT)
                base_sz = int(BASE_SZ_U_TIGHT)
            else:
                half = int(HALF)
                base_sz = int(BASE_SZ)

            bid_px = int(round(fair - half))
            ask_px = int(round(fair + half))
            bid_px = min(bid_px, ba - 1)
            ask_px = max(ask_px, bb + 1)
            if ask_px <= bid_px:
                ask_px = bid_px + 1

            lim = LIMITS[sym]
            pv = int(pos.get(sym, 0))

            q_buy = min(base_sz + (BOOST_BID if sym == U and m67_buy else 0), lim - pv)
            q_sell = min(base_sz, lim + pv)
            if sym == U and m38_buy:
                q_sell = 0
            if sym == U and m55_sell:
                q_buy = 0

            ol: list[Order] = []
            if q_buy > 0 and bid_px > 0:
                ol.append(Order(sym, bid_px, q_buy))
            if q_sell > 0:
                ol.append(Order(sym, ask_px, -q_sell))
            if ol:
                out[sym] = ol

        return out, 0, json.dumps(store, separators=(",", ":"))
