"""
Round 4 — **v7 (Mark67 × Sonic U) + Phase-2 Burst-B VEV_5300** addon.

**U leg:** identical to `trader_v7.py` (improved bid/ask exits, gate-off improved-ask liquidation).

**VEV_5300 leg (Burst-B, same definition as `phase2_r4_analysis.py`):**
At this `timestamp`, across **all** `market_trades` symbols: **≥ 3** trade rows, at least one with
buyer **Mark 01** and seller **Mark 22**, and **≥ 3** distinct symbols matching `VEV_*`.
When **Sonic tight** (same as U) and **U spread ≤ 6**, buy **VEV_5300** with the same execution
pattern as U entries (`improved_bid` if spread ≥ 2 else `ASK_LIFT_Q` at ask). Flatten long
**VEV_5300** when gate opens (improved ask if spread ≥ 2 else bid). Cap **Q5300 = 4** per signal,
position limit **300**.

Sparse overlap with Mark67 (see `analysis_outputs/r8_m67_sonic_burstB_overlap.txt`).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
K5200 = "VEV_5200"
K5300 = "VEV_5300"
MARK67 = "Mark 67"
MARK01 = "Mark 01"
MARK22 = "Mark 22"

LIMIT_U = 200
LIMIT_V5300 = 300
TH = 2
MAX_U_SPREAD = 6

ENTRY_Q = 5
EXIT_Q = 5
ASK_LIFT_Q = 1
Q5300 = 4
EMA_ALPHA = 0.06
PROFIT = 1.05

_KEY_EMA = "_ema_mid"
_KEY_INIT = "_inited"
_KEY_LAST_ENTRY_TS = "_last_m67_entry_ts"
_KEY_LAST_BURST_TS = "_last_burst5300_ts"


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _bb_ba(d: OrderDepth | None) -> tuple[int | None, int | None]:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders), min(d.sell_orders)


def _spr(d: OrderDepth | None) -> int | None:
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def _improved_bid(bb: int, ba: int) -> int:
    if ba - bb >= 2:
        return min(bb + 1, ba - 1)
    return bb


def _improved_ask(bb: int, ba: int) -> int:
    if ba - bb >= 2:
        return max(bb + 1, ba - 1)
    return ba


def _joint_tight(depths: dict[str, OrderDepth]) -> bool:
    s2 = _spr(depths.get(K5200))
    s3 = _spr(depths.get(K5300))
    if s2 is None or s3 is None:
        return False
    return s2 <= TH and s3 <= TH


def _mark67_buy_agg_u(state: TradingState, ba_u: int) -> bool:
    for tr in state.market_trades.get(U, []):
        buyer = getattr(tr, "buyer", None)
        price = getattr(tr, "price", None)
        if buyer != MARK67 or price is None:
            continue
        if int(price) >= int(ba_u):
            return True
    return False


def _burst_b_vev5300(state: TradingState) -> bool:
    """Mark01→Mark22 basket: ≥3 trades total at this timestamp, ≥3 distinct VEV_* symbols."""
    rows: list[tuple[str, str, str]] = []
    for sym, lst in state.market_trades.items():
        for tr in lst:
            buyer = getattr(tr, "buyer", None)
            seller = getattr(tr, "seller", None)
            s = getattr(tr, "symbol", sym)
            if buyer is None or seller is None:
                continue
            rows.append((str(s), str(buyer), str(seller)))
    if len(rows) < 3:
        return False
    if not any(b == MARK01 and s == MARK22 for _, b, s in rows):
        return False
    vev = {s for s, _, _ in rows if s.startswith("VEV_")}
    return len(vev) >= 3


def _flatten_long(
    orders: dict[str, list[Order]], sym: str, depths: dict[str, OrderDepth], pos: int
) -> None:
    if pos <= 0:
        return
    d = depths.get(sym)
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return
    spr = int(ba - bb)
    if spr >= 2:
        orders.setdefault(sym, []).append(Order(sym, _improved_ask(bb, ba), -pos))
    else:
        orders.setdefault(sym, []).append(Order(sym, bb, -pos))


def _maybe_buy_improved(
    orders: dict[str, list[Order]],
    sym: str,
    depths: dict[str, OrderDepth],
    pos: int,
    limit: int,
    q_cap: int,
) -> bool:
    """Return True if a buy order was placed."""
    d = depths.get(sym)
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return False
    spr = int(ba - bb)
    q = min(q_cap, limit - pos)
    if q <= 0:
        return False
    if spr >= 2:
        buy_px = _improved_bid(bb, ba)
        if buy_px >= ba:
            return False
        orders.setdefault(sym, []).append(Order(sym, buy_px, q))
        return True
    if spr == 1:
        orders.setdefault(sym, []).append(Order(sym, ba, min(ASK_LIFT_Q, q)))
        return True
    return False


class Trader:
    def run(self, state: TradingState):
        store = _parse_td(state.traderData)
        depths = state.order_depths

        du = depths.get(U)
        bb_u, ba_u = _bb_ba(du)
        if bb_u is None or ba_u is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        u_spread = int(ba_u - bb_u)
        if u_spread > MAX_U_SPREAD:
            tight = False
        else:
            tight = _joint_tight(depths)

        mid = 0.5 * float(bb_u + ba_u)
        ts = int(state.timestamp)

        if not store.get(_KEY_INIT):
            store[_KEY_INIT] = True
            store[_KEY_EMA] = mid
            store[_KEY_LAST_ENTRY_TS] = -1
            store[_KEY_LAST_BURST_TS] = -1
        else:
            ema = float(store.get(_KEY_EMA, mid))
            ema = EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * ema
            store[_KEY_EMA] = ema

        pos_u = int(state.position.get(U, 0))
        pos_v = int(state.position.get(K5300, 0))
        orders: dict[str, list[Order]] = {}

        if not tight:
            _flatten_long(orders, U, depths, pos_u)
            _flatten_long(orders, K5300, depths, pos_v)
            if pos_u < 0:
                orders.setdefault(U, []).append(Order(U, ba_u, -pos_u))
            out = {k: v for k, v in orders.items() if v}
            return out, 0, json.dumps(store, separators=(",", ":"))

        if pos_u > 0 and mid > ema + PROFIT:
            q = min(EXIT_Q, pos_u)
            if q > 0:
                sell_px = _improved_ask(bb_u, ba_u)
                orders.setdefault(U, []).append(Order(U, sell_px, -q))

        sig = _mark67_buy_agg_u(state, ba_u)
        last_ent = int(store.get(_KEY_LAST_ENTRY_TS, -1))
        if sig and pos_u < LIMIT_U and ts != last_ent:
            if _maybe_buy_improved(orders, U, depths, pos_u, LIMIT_U, ENTRY_Q):
                store[_KEY_LAST_ENTRY_TS] = ts

        burst = _burst_b_vev5300(state)
        last_b = int(store.get(_KEY_LAST_BURST_TS, -1))
        if burst and pos_v < LIMIT_V5300 and ts != last_b:
            if _maybe_buy_improved(orders, K5300, depths, pos_v, LIMIT_V5300, Q5300):
                store[_KEY_LAST_BURST_TS] = ts

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
