"""
Round 4 — **v3 + aggressive tight-regime MM** (tape: population extract K=20 fwd much better when gate tight).

- Same structure as **trader_v3**: extract + hydro + VEV_5200/5300 **only** when Sonic tight; other VEVs always;
  Mark 38 / Mark 67 hooks on extract unchanged.
- **Tight regime** (both VEV_5200 and VEV_5300 BBO spread <= 2): quote **mid ± 1** (was ±2) and raise sizes
  (underlying/hydro **12**, VEVs **9** vs baseline **6**) to push integer-tick fill probability while gate says
  extract path is favorable.
- **Wide regime** (non-gate VEVs only): keep **mid ± 2**, size **6** — unchanged vs v3.
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

HALF_WIDE = 2
HALF_TIGHT = 1
BASE_WIDE = 6
BASE_TIGHT_UH = 12
BASE_TIGHT_VEV = 9
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


def extract_trade_flags(state: TradingState) -> tuple[bool, bool]:
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

        tight = sonic_tight(depths)
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

            if sym in (U, H) and not tight:
                continue
            if sym in (G520, G530) and not tight:
                continue

            if tight:
                half = int(HALF_TIGHT)
                if sym in (U, H):
                    base_sz = int(BASE_TIGHT_UH)
                else:
                    base_sz = int(BASE_TIGHT_VEV)
            else:
                half = int(HALF_WIDE)
                base_sz = int(BASE_WIDE)

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

            ol: list[Order] = []
            if q_buy > 0 and bid_px > 0:
                ol.append(Order(sym, bid_px, q_buy))
            if q_sell > 0:
                ol.append(Order(sym, ask_px, -q_sell))
            if ol:
                out[sym] = ol

        return out, 0, json.dumps(store, separators=(",", ":"))
