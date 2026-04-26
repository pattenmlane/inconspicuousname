"""
Round 4 — **Mark 67** on-tape **buyer-aggressive** prints on **U** × **Sonic joint gate** (passive entry).

Tape analysis (Phases 1–3): **Mark 67** aggressive buys on **VELVETFRUIT_EXTRACT** show positive
short-horizon forward mid; effect is stronger when **VEV_5200/5300** spreads are both **≤ 2**
(Phase 3 interaction table).

Requires **backtester** `TestRunner` to populate `state.market_trades` **before** `Trader.run()`
(same-timestamp tape); see `imc-prosperity-4-backtester/prosperity4bt/test_runner.py`.

Logic (U only):
- `tight` = Sonic gate + U top spread ≤ `MAX_U_SPREAD`.
- **Signal:** any `market_trades[U]` row with `buyer == "Mark 67"` and `price >= ask1` (buy-aggressive).
- **Entry:** on signal while `tight`, at most once per timestamp: if U spread **≥ 2**, **improved bid** for `ENTRY_Q`; if spread **== 1** (common in Mark67∩Sonic slice), small **ask lift** `ASK_LIFT_Q` so `--match-trades worse` can fill.
- **Exit:** EMA(mid) take-profit (partial at ask) while tight; **flatten** when not tight.
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

ENTRY_Q = 8
EXIT_Q = 8
ASK_LIFT_Q = 3
EMA_ALPHA = 0.06
PROFIT = 0.55

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


def _improved_bid(bb: int, ba: int) -> int:
    if ba - bb >= 2:
        return min(bb + 1, ba - 1)
    return bb


def _joint_tight(depths: dict[str, OrderDepth]) -> bool:
    s2 = _spr(depths.get(K5200))
    s3 = _spr(depths.get(K5300))
    if s2 is None or s3 is None:
        return False
    return s2 <= TH and s3 <= TH


def _mark67_buy_agg_u(state: TradingState, ba_u: int) -> bool:
    # Duck-type trades: avoid `isinstance(..., Trade)` — import paths can bind a
    # different `Trade` class than the backtester used when building `market_trades`.
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
                orders[U].append(Order(U, bb_u, -pos))
            elif pos < 0:
                orders[U].append(Order(U, ba_u, -pos))
            out = {k: v for k, v in orders.items() if v}
            return out, 0, json.dumps(store, separators=(",", ":"))

        if pos > 0 and mid > ema + PROFIT:
            q = min(EXIT_Q, pos)
            if q > 0:
                orders[U].append(Order(U, ba_u, -q))

        sig = _mark67_buy_agg_u(state, ba_u)
        last_ent = int(store.get(_KEY_LAST_ENTRY_TS, -1))
        if sig and pos < LIMIT_U and ts != last_ent:
            spr_u = int(ba_u - bb_u)
            if spr_u >= 2:
                buy_px = _improved_bid(bb_u, ba_u)
                if buy_px < ba_u:
                    q = min(ENTRY_Q, LIMIT_U - pos)
                    if q > 0:
                        orders[U].append(Order(U, buy_px, q))
                        store[_KEY_LAST_ENTRY_TS] = ts
            elif spr_u == 1:
                q = min(ASK_LIFT_Q, LIMIT_U - pos)
                if q > 0:
                    orders[U].append(Order(U, ba_u, q))
                    store[_KEY_LAST_ENTRY_TS] = ts

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
