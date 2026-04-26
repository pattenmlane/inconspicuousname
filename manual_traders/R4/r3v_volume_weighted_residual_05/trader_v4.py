"""
Round 4 — **Sonic joint gate** + **tight U** + **passive extract** (response to v3 failure).

Phase 3: pooled extract fwd@20 is higher when VEV_5200/5300 spreads are both <= 2.
Phase 2: Mark67-style tape signals were defined on **tight U** (spread <= 6); v3 showed
**aggressive** buy-at-ask on gate **edges** loses badly under `--match-trades worse`.

This version (U only; no hydrogel / no VEV orders):
- `tight` = Sonic (s5200<=2, s5300<=2) AND U spread <= `MAX_U_SPREAD`.
- Maintain EMA(mid) in `traderData`.
- **Entry:** while `tight`, `mid < ema - DIP`, and room under `LIMIT_U`, post **best bid**
  for `ENTRY_Q` (passive lift from sellers / book).
- **Take profit:** while `tight` and long, if `mid > ema + PROFIT`, sell `EXIT_Q` at **best ask**
  (capped by position).
- **Risk off:** when not `tight`, flatten long at **best bid**.

Note: `TestRunner` now attaches **raw** same-timestamp tape to `state.market_trades` before
`run()` (see `imc-prosperity-4-backtester/.../test_runner.py`); this trader still does not
use counterparty IDs (BBO + spreads only).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
K5200 = "VEV_5200"
K5300 = "VEV_5300"

LIMIT_U = 200
TH = 2
MAX_U_SPREAD = 6

ENTRY_Q = 10
EXIT_Q = 10
EMA_ALPHA = 0.05
DIP = 0.35
PROFIT = 0.5

_KEY_EMA = "_ema_mid"
_KEY_INIT = "_inited"


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
    """Price for passive buys under `--match-trades worse`.

    Equality at the touch bid does **not** match tape prints; need a strictly higher
    limit so trades at/near the bid can fill when they print below our limit.
    """
    if ba - bb >= 2:
        return min(bb + 1, ba - 1)
    return bb


def _joint_tight(depths: dict[str, OrderDepth]) -> bool:
    s2 = _spr(depths.get(K5200))
    s3 = _spr(depths.get(K5300))
    if s2 is None or s3 is None:
        return False
    return s2 <= TH and s3 <= TH


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

        if not store.get(_KEY_INIT):
            store[_KEY_INIT] = True
            store[_KEY_EMA] = mid
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ema = float(store.get(_KEY_EMA, mid))
        ema = EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * ema
        store[_KEY_EMA] = ema

        pos = int(state.position.get(U, 0))
        orders: dict[str, list[Order]] = {U: []}

        if not tight:
            if pos > 0:
                orders[U].append(Order(U, bb_u, -pos))
            elif pos < 0:
                orders[U].append(Order(U, ba_u, -pos))
            out = {k: v for k, v in orders.items() if v}
            return out, 0, json.dumps(store, separators=(",", ":"))

        # tight regime
        if pos > 0 and mid > ema + PROFIT:
            q = min(EXIT_Q, pos)
            if q > 0:
                orders[U].append(Order(U, ba_u, -q))
        buy_px = _improved_bid(bb_u, ba_u)
        if pos < LIMIT_U and mid < ema - DIP and buy_px < ba_u:
            q = min(ENTRY_Q, LIMIT_U - pos)
            if q > 0:
                orders[U].append(Order(U, buy_px, q))

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
