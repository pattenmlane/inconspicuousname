"""
Round 4 — **v7 + skip 1-tick U entries** (Mark 67 × Sonic).

Parent: `trader_v7.py`.

**Change:** do **not** open new Mark67-driven longs when **U** top-of-book spread is **1**
(v7 used **ask lift** at touch there). When **U spread ≥ 2**, enter at **best ask** (`ba_u`)
so the order matches the **book** first (same as paying touch on a multi-tick book). Rationale:
`improved_bid` alone often sits **below** Mark67’s aggressive print at the ask, so under
`--match-trades worse` there is **no** fill (tape equality does not match; no resting sell
inside spread).

Exits, EMA take-profit, and gate-off liquidation unchanged from v7.
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
K5200 = "VEV_5200"
K5300 = "VEV_5300"
MARK67 = "Mark 67"

LIMIT_U = 200
TH = 2
MAX_U_SPREAD = 6

ENTRY_Q = 5
EXIT_Q = 5
EMA_ALPHA = 0.06
PROFIT = 1.05

_KEY_EMA = "_ema_mid"
_KEY_INIT = "_inited"
_KEY_LAST_ENTRY_TS = "_last_m67_entry_ts"


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
        else:
            ema = float(store.get(_KEY_EMA, mid))
            ema = EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * ema
            store[_KEY_EMA] = ema

        pos = int(state.position.get(U, 0))
        orders: dict[str, list[Order]] = {U: []}

        if not tight:
            if pos > 0:
                if u_spread >= 2:
                    px = _improved_ask(bb_u, ba_u)
                    orders[U].append(Order(U, px, -pos))
                else:
                    orders[U].append(Order(U, bb_u, -pos))
            elif pos < 0:
                orders[U].append(Order(U, ba_u, -pos))
            out = {k: v for k, v in orders.items() if v}
            return out, 0, json.dumps(store, separators=(",", ":"))

        if pos > 0 and mid > ema + PROFIT:
            q = min(EXIT_Q, pos)
            if q > 0:
                sell_px = _improved_ask(bb_u, ba_u)
                orders[U].append(Order(U, sell_px, -q))

        sig = _mark67_buy_agg_u(state, ba_u)
        last_ent = int(store.get(_KEY_LAST_ENTRY_TS, -1))
        if sig and pos < LIMIT_U and ts != last_ent:
            spr_u = int(ba_u - bb_u)
            if spr_u >= 2:
                q = min(ENTRY_Q, LIMIT_U - pos)
                if q > 0:
                    orders[U].append(Order(U, ba_u, q))
                    store[_KEY_LAST_ENTRY_TS] = ts

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
