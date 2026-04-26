"""
v27 — `vouchers_final_strategy` (STRATEGY.txt / ORIGINAL_DISCORD_QUOTES.txt):

- **Sonic gate:** `VEV_5200` and `VEV_5300` top-of-book spread both **<= 2** at this timestamp.
- **Regime off:** flatten `VELVETFRUIT_EXTRACT` + all VEV at touch (no new flow).
- **Regime on:** two-sided **touch** market-making (bid at best bid, ask at best ask) on
  `VELVETFRUIT_EXTRACT` and on **VEV_5200** / **VEV_5300** (spreads are the signal; quotes sit in
  the book state the analysis uses). **Inventory skew** (Frankfurt-style): shift both bid and
  ask by a few ticks when inventory builds so we do not one-sidedly stack.
- **No HYDROGEL_PACK.**
"""
from __future__ import annotations

import json
import math
from typing import Any

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
K5200, K5300 = "VEV_5200", "VEV_5300"
FOCUS = (K5200, K5300)
VOU = [f"VEV_{k}" for k in STRIKES]
LIM: dict[str, int] = {U: 200, **{f"VEV_{k}": 300 for k in STRIKES}}

TH = 2
Q_U = 6
Q_V = 4
SKEW_U = 0.05
SKEW_V = 0.04
EMA_ALPHA = 0.12
MAX_SPR = 8

_CSV = "csv_day"
_EMA = "ema_U"
_ANC = {5250.0: 0, 5245.0: 1, 5267.5: 2}


def _td(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        o = json.loads(s)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _bb_ba(d: OrderDepth) -> tuple[int | None, int | None]:
    if not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders), min(d.sell_orders)


def _spr1(d: OrderDepth) -> int | None:
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def _day(state: TradingState, store: dict[str, Any]) -> int:
    if _CSV in store:
        return int(store[_CSV])
    du = state.order_depths.get(U)
    if du is None:
        return 0
    bb, ba = _bb_ba(du)
    if bb is None or ba is None:
        return 0
    s0 = 0.5 * (bb + ba)
    best_d, best_e = 0, 1e9
    for s_anchor, day_i in _ANC.items():
        e = abs(s0 - s_anchor)
        if e < best_e:
            best_e, best_d = e, int(day_i)
    if best_e > 5.0:
        best_d = 0
    store[_CSV] = best_d
    return best_d


def _ema(store: dict[str, Any], m: float) -> float:
    p = store.get(_EMA)
    if p is not None and isinstance(p, (int, float)) and math.isfinite(float(p)):
        e = EMA_ALPHA * m + (1.0 - EMA_ALPHA) * float(p)
    else:
        e = m
    store[_EMA] = e
    return e


def _gate(st: TradingState) -> bool:
    d2, d3 = st.order_depths.get(K5200), st.order_depths.get(K5300)
    if d2 is None or d3 is None:
        return False
    a, b = _spr1(d2), _spr1(d3)
    if a is None or b is None:
        return False
    return a <= TH and b <= TH


def _sk(pos: int, k: float) -> int:
    t = int(round(k * float(pos)))
    if t > 2:
        t = 2
    if t < -2:
        t = -2
    return t


def _mm_sym(sym: str, d: OrderDepth, pos: int, lim: int, q0: int, skew: float) -> list[Order]:
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return []
    w = int(ba - bb)
    if w < 1 or w > MAX_SPR:
        return []
    st = _sk(pos, skew)
    bpx = bb - st
    apx = ba - st
    if bpx >= apx:
        bpx, apx = bb, ba
    out: list[Order] = []
    # buy (positive order): can hold up to +lim
    q_b = min(q0, max(0, lim - pos))
    if q_b > 0 and bpx < ba:
        out.append(Order(sym, bpx, q_b))
    # sell (negative order): from pos can go to -lim
    q_s = min(q0, max(0, pos + lim))
    if q_s > 0 and apx > bb:
        out.append(Order(sym, apx, -q_s))
    return out


def _flat(st: TradingState) -> dict[str, list[Order]]:
    o: dict[str, list[Order]] = {U: []}
    for v in VOU:
        o[v] = []
    for sym in [U, *VOU]:
        p = int(st.position.get(sym, 0))
        if p == 0:
            continue
        d = st.order_depths.get(sym)
        if d is None:
            continue
        bb, ba = _bb_ba(d)
        if bb is None or ba is None:
            continue
        l = LIM[sym]
        if p > 0:
            o[sym] = [Order(sym, ba, -min(p, l))]
        else:
            o[sym] = [Order(sym, bb, min(-p, l))]
    return o


class Trader:
    def run(self, state: TradingState):
        store = _td(state.traderData)
        _ = _day(state, store)
        du = state.order_depths.get(U)
        if du is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        bbu, bau = _bb_ba(du)
        if bbu is None or bau is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        m = 0.5 * (bbu + bau)
        _ = _ema(store, m)

        if not _gate(state):
            f = _flat(state)
            return {k: v for k, v in f.items() if v}, 0, json.dumps(store, separators=(",", ":"))

        ords: dict[str, list[Order]] = {U: [], **{v: [] for v in VOU}}
        pu = int(state.position.get(U, 0))
        ords[U] = _mm_sym(U, du, pu, LIM[U], Q_U, SKEW_U)
        for sym in FOCUS:
            d = state.order_depths.get(sym)
            if d is None:
                continue
            p = int(state.position.get(sym, 0))
            ords[sym] = _mm_sym(sym, d, p, LIM[sym], Q_V, SKEW_V)
        return {k: v for k, v in ords.items() if v}, 0, json.dumps(store, separators=(",", ":"))
