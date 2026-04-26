"""
Round 4 — counterparty-aware baseline (Phase 2 execution hook).

- Default: simple mid \u00b1 2 MM on HYDROGEL_PACK, VELVETFRUIT_EXTRACT, all VEV_* (size 6).
- **Adverse selection (Phase 1):** if `TradingState.market_trades` at this tick includes an
  aggressive **Mark 38 buy** on extract (buyer==Mark 38, trade price at/above ask), **omit extract ask**
  for this step (still bid) \u2014 Phase 1 showed Mark 38 buy_agg with most negative K=20 markouts.
- **Optional lean (Phase 1):** if **Mark 67** aggressive **buy** on extract at this tick, add +2 to bid size
  (cap at limit).

TTE not modeled in fair; width fixed. Position limits per round4description.
"""
from __future__ import annotations

import json
import math
from typing import Any

from datamodel import Order, OrderDepth, TradingState

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEVS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
LIMITS = {H: 200, U: 200, **{v: 300 for v in VEVS}}

HALF = 2
BASE_SZ = 6
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


def extract_trade_flags(state: TradingState) -> tuple[bool, bool]:
    """Returns (mark38_aggressive_buy, mark67_aggressive_buy) on extract this tick."""
    mt = getattr(state, "market_trades", None) or {}
    trades = mt.get(U)
    if not trades:
        return False, False
    depths = getattr(state, "order_depths", {}) or {}
    du = depths.get(U)
    if du is None:
        return False, False
    bb, ba = best_bid_ask(du)
    if bb is None or ba is None:
        return False, False
    m38 = m67 = False
    for t in trades:
        buyer = getattr(t, "buyer", None)
        pr = int(getattr(t, "price", 0))
        if buyer == "Mark 38" and pr >= ba:
            m38 = True
        if buyer == "Mark 67" and pr >= ba:
            m67 = True
    return m38, m67


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

        m38_buy, m67_buy = extract_trade_flags(state)

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
            bid_px = int(round(fair - HALF))
            ask_px = int(round(fair + HALF))
            bid_px = min(bid_px, ba - 1)
            ask_px = max(ask_px, bb + 1)
            if ask_px <= bid_px:
                ask_px = bid_px + 1

            lim = LIMITS[sym]
            pv = int(pos.get(sym, 0))
            q_buy = min(BASE_SZ + (BOOST_BID if sym == U and m67_buy else 0), lim - pv)
            q_sell = min(BASE_SZ, lim + pv)
            if sym == U and m38_buy:
                q_sell = 0

            ol: list[Order] = []
            if q_buy > 0 and bid_px > 0:
                ol.append(Order(sym, bid_px, q_buy))
            if q_sell > 0:
                ol.append(Order(sym, ask_px, -q_sell))
            if ol:
                out[sym] = ol

        return out, 0, json.dumps(store, separators=(",", ":"))
