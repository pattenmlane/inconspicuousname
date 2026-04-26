"""
Round 4 — **v19 + Mark 55 aggressive-sell widen extract ask** (Phase 1 adverse-selection; orthogonal to M67).

- **v19**: v12 schedule + M67 **+1** ask (cap 4) + M67 wide bid **mid\u22122** on lift ticks.
- **v20**: same, plus when **Sonic tight** and **Mark 55** **aggressive sell** on extract (`seller==Mark 55`, `price<=bid`),
  add **+1** to tight **ask** half (combined M67+M55 bumps **capped at 5** so both can fire without double-counting past cap).
- **v6** tried **zero bid** on M55 sell and failed; this tests **passive ask retreat** only (inventory / sell-pressure side).
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
G520 = "VEV_5200"
G530 = "VEV_5300"
GATE_MAX = 2

LIMITS = {H: 200, U: 200, **{v: 300 for v in VEVS}}

HALF = 2
HALF_U_TIGHT_BID = 1
HALF_U_TIGHT_BID_M67 = 2
HALF_U_TIGHT_ASK_D12 = 3
HALF_U_TIGHT_ASK_D3 = 2
HALF_U_TIGHT_ASK_MAX = 5
R4_DAY_OPEN_MID: list[tuple[int, float]] = [(1, 5245.0), (2, 5267.5), (3, 5295.5)]
MID_MATCH_TOL = 1.25
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
    """m38 aggressive buy, m67 aggressive buy, m55 aggressive sell on extract."""
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


def infer_tape_r4_day(store: dict[str, Any], depths: dict) -> tuple[int, dict[str, Any]]:
    existing = store.get("tape_r4_day")
    if isinstance(existing, int) and 1 <= existing <= 3:
        return int(existing), dict(store)
    du = depths.get(U)
    mid0 = micro_mid(du) if du is not None else None
    day = 1
    if mid0 is not None and math.isfinite(float(mid0)):
        m = float(mid0)
        best_d, best_err = 1, 1e9
        for d, ref in R4_DAY_OPEN_MID:
            err = abs(m - ref)
            if err < best_err:
                best_err = err
                best_d = d
        if best_err <= MID_MATCH_TOL:
            day = int(best_d)
    out = dict(store)
    out["tape_r4_day"] = int(day)
    return int(day), out


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

        tape_day, store = infer_tape_r4_day(store, depths)

        tight = sonic_tight(depths)
        m38_buy, m67_buy, m55_sell = extract_trade_flags(state)

        base_ask = int(HALF_U_TIGHT_ASK_D3) if tape_day >= 3 else int(HALF_U_TIGHT_ASK_D12)
        if tight:
            extra = 0
            if m67_buy:
                extra += 1
            if m55_sell:
                extra += 1
            base_ask = min(base_ask + extra, int(HALF_U_TIGHT_ASK_MAX))

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
                half_b = int(HALF_U_TIGHT_BID_M67) if m67_buy else int(HALF_U_TIGHT_BID)
                half_a = int(base_ask)
                base_sz = int(BASE_SZ_U_TIGHT)
                bid_px = int(round(fair - half_b))
                ask_px = int(round(fair + half_a))
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

            ol: list[Order] = []
            if q_buy > 0 and bid_px > 0:
                ol.append(Order(sym, bid_px, q_buy))
            if q_sell > 0:
                ol.append(Order(sym, ask_px, -q_sell))
            if ol:
                out[sym] = ol

        return out, 0, json.dumps(store, separators=(",", ":"))
