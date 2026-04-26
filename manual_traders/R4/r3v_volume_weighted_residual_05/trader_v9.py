"""
Round 4 — **v7 + Phase-1 Mark 22 passive seller on U** (second counterparty leg).

**U leg A (unchanged from v7):** Mark **67** buyer-aggressive on **VELVETFRUIT_EXTRACT**
(`price >= ask1`), **Sonic** joint gate, **U** spread ≤ 6, improved bid / ask-lift entry,
EMA take-profit, **improved-ask** liquidation when gate opens (spread ≥ 2).

**U leg B (new):** Phase-1 candidate — **Mark 22** as **passive seller** on **U**:
any tape row with `seller == "Mark 22"`, symbol **U**, `price <= bid1` (sell-aggressive from
liquidity provider side = passive sell). Same gate and **same** entry/exit machinery as leg A,
**separate** per-timestamp dedupe key (`_last_m22_entry_ts`). Smaller clip **`ENTRY_Q_M22 = 3`**
so overlap with Mark67 does not ramp size too fast.

Tape frequency under this gate is **very low** (see `analysis_outputs/r9_m22_passive_u_sonic_overlap.txt`).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
K5200 = "VEV_5200"
K5300 = "VEV_5300"
MARK67 = "Mark 67"
MARK22 = "Mark 22"

LIMIT_U = 200
TH = 2
MAX_U_SPREAD = 6

ENTRY_Q = 5
ENTRY_Q_M22 = 3
EXIT_Q = 5
ASK_LIFT_Q = 1
EMA_ALPHA = 0.06
PROFIT = 1.05

_KEY_EMA = "_ema_mid"
_KEY_INIT = "_inited"
_KEY_LAST_ENTRY_TS = "_last_m67_entry_ts"
_KEY_LAST_M22_TS = "_last_m22_entry_ts"


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


def _mark22_passive_sell_u(state: TradingState, bb_u: int) -> bool:
    for tr in state.market_trades.get(U, []):
        seller = getattr(tr, "seller", None)
        price = getattr(tr, "price", None)
        if seller != MARK22 or price is None:
            continue
        if int(price) <= int(bb_u):
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
            store[_KEY_LAST_M22_TS] = -1
        else:
            ema = float(store.get(_KEY_EMA, mid))
            ema = EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * ema
            store[_KEY_EMA] = ema

        pos = int(state.position.get(U, 0))
        orders: dict[str, list[Order]] = {U: []}

        if not tight:
            if pos > 0:
                if u_spread >= 2:
                    orders[U].append(Order(U, _improved_ask(bb_u, ba_u), -pos))
                else:
                    orders[U].append(Order(U, bb_u, -pos))
            elif pos < 0:
                orders[U].append(Order(U, ba_u, -pos))
            out = {k: v for k, v in orders.items() if v}
            return out, 0, json.dumps(store, separators=(",", ":"))

        if pos > 0 and mid > ema + PROFIT:
            q = min(EXIT_Q, pos)
            if q > 0:
                orders[U].append(Order(U, _improved_ask(bb_u, ba_u), -q))

        last_m67 = int(store.get(_KEY_LAST_ENTRY_TS, -1))
        if _mark67_buy_agg_u(state, ba_u) and pos < LIMIT_U and ts != last_m67:
            spr_u = int(ba_u - bb_u)
            if spr_u >= 2:
                buy_px = _improved_bid(bb_u, ba_u)
                if buy_px < ba_u:
                    q = min(ENTRY_Q, LIMIT_U - pos)
                    if q > 0:
                        orders[U].append(Order(U, buy_px, q))
                        store[_KEY_LAST_ENTRY_TS] = ts
                        pos += q
            elif spr_u == 1:
                q = min(ASK_LIFT_Q, LIMIT_U - pos)
                if q > 0:
                    orders[U].append(Order(U, ba_u, q))
                    store[_KEY_LAST_ENTRY_TS] = ts
                    pos += q

        last_m22 = int(store.get(_KEY_LAST_M22_TS, -1))
        if _mark22_passive_sell_u(state, bb_u) and pos < LIMIT_U and ts != last_m22:
            spr_u = int(ba_u - bb_u)
            if spr_u >= 2:
                buy_px = _improved_bid(bb_u, ba_u)
                if buy_px < ba_u:
                    q = min(ENTRY_Q_M22, LIMIT_U - pos)
                    if q > 0:
                        orders[U].append(Order(U, buy_px, q))
                        store[_KEY_LAST_M22_TS] = ts
            elif spr_u == 1:
                q = min(ASK_LIFT_Q, LIMIT_U - pos)
                if q > 0:
                    orders[U].append(Order(U, ba_u, q))
                    store[_KEY_LAST_M22_TS] = ts

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
