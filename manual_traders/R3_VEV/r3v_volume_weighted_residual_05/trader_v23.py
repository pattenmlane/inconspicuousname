"""
Round 3 — vouchers_final_strategy only (see round3work/vouchers_final_strategy/STRATEGY.txt).

- **Sonic / inclineGod:** Regime = VEV_5200 and VEV_5300 top-of-book spread both <= **TH=2** (same
  tick units as the tape: ask1 - bid1). When **not** in this joint-tight regime, we **do not**
  open new risk in extract or VEV; we only **flatten** inventory at touch to respect “wide
  book = execution dominates”.

- **Tight + directional extract (optional reading of K-step forward-mid analysis):** when the
  gate is **on**, EMA(microstructure mid of extract) mean-reversion: buy extract when
  mid << EMA, sell / reduce long when mid >> EMA. Same gate used to **add** long exposure on
  **VEV_5200** and **VEV_5300** at the ask (short-hold “flow” in the two strikes called out in
  Discord) when the extract signal agrees (mid below EMA by at least **U_EDGE** ticks).

- **No HYDROGEL_PACK** — PnL objective is extract + VEVs per task scope.

- **DTE / timing:** dte_open = 8 - csv_day; dte_eff = dte_open - (timestamp//100)/10000 (round3description).
  Anchors for csv_day: same extract mid anchors as other Round-3 VEV scripts.
"""
from __future__ import annotations

import json
import math
from typing import Any

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
VOU = [f"VEV_{k}" for k in STRIKES]

# Position limits: round3description
LIMITS: dict[str, int] = {U: 200, **{f"VEV_{k}": 300 for k in STRIKES}}

TH = 2
MAX_VEV_SPREAD = 8

EMA_U_ALPHA = 0.08
U_EDGE = 2  # ticks from EMA to act
U_Q = 18
VEV_FLOW_Q = 24

_ANCHOR_S0 = {5250.0: 0, 5245.0: 1, 5267.5: 2}

_CSV = "csv_day"
_EMA_U = "ema_U_mid"
def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def dte_open(csv_day: int) -> float:
    return float(8 - csv_day)


def dte_eff(csv_day: int, ts: int) -> float:
    return max(dte_open(csv_day) - (int(ts) // 100) / 10_000.0, 1e-6)


def _bb_ba(d: OrderDepth) -> tuple[int | None, int | None]:
    if not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders), min(d.sell_orders)


def spread1(d: OrderDepth) -> int | None:
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def infer_csv_day(state: TradingState, store: dict[str, Any]) -> int:
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
    for a, d in _ANCHOR_S0.items():
        e = abs(s0 - a)
        if e < best_e:
            best_e, best_d = e, d
    if best_e > 5.0:
        best_d = 0
    store[_CSV] = best_d
    return best_d


def _update_ema_u(store: dict[str, Any], mid: float) -> float:
    prev = store.get(_EMA_U)
    if prev is not None and isinstance(prev, (int, float)) and math.isfinite(float(prev)):
        e = EMA_U_ALPHA * mid + (1.0 - EMA_U_ALPHA) * float(prev)
    else:
        e = mid
    store[_EMA_U] = e
    return e


def _joint_tight(state: TradingState) -> bool:
    d2 = state.order_depths.get(VEV_5200)
    d3 = state.order_depths.get(VEV_5300)
    if d2 is None or d3 is None:
        return False
    s2, s3 = spread1(d2), spread1(d3)
    if s2 is None or s3 is None:
        return False
    return s2 <= TH and s3 <= TH


def _flatten_one(
    sym: str, pos: int, d: OrderDepth, lim: int
) -> list[Order]:
    if pos == 0 or d is None:
        return []
    bb, ba = _bb_ba(d)
    if bb is None or ba is None:
        return []
    if pos > 0:
        q = min(pos, lim)
        return [Order(sym, ba, -q)]
    q = min(-pos, lim)
    return [Order(sym, bb, q)]


class Trader:
    def run(self, state: TradingState):
        store = _parse_td(state.traderData)
        csv_d = infer_csv_day(state, store)

        _ = dte_eff(csv_d, state.timestamp)  # available for future IV work

        du = state.order_depths.get(U)
        if du is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        bb_u, ba_u = _bb_ba(du)
        if bb_u is None or ba_u is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        s_mid = 0.5 * (bb_u + ba_u)
        ema = _update_ema_u(store, s_mid)
        dev = s_mid - ema
        u_spr = int(ba_u - bb_u)

        gate = _joint_tight(state)
        orders: dict[str, list[Order]] = {U: []}
        for v in VOU:
            orders[v] = []

        if not gate:
            p_u = int(state.position.get(U, 0))
            if p_u > 0:
                orders[U] += _flatten_one(U, p_u, du, 200)
            elif p_u < 0:
                orders[U] += _flatten_one(U, p_u, du, 200)
            for v in VOU:
                p = int(state.position.get(v, 0))
                dd = state.order_depths.get(v)
                orders[v] += _flatten_one(v, p, dd, 300)
            out = {k: v for k, v in orders.items() if v}
            return out, 0, json.dumps(store, separators=(",", ":"))

        # --- risk-on: joint tight gate (Sonic) ---
        pos_u = int(state.position.get(U, 0))
        if u_spr <= 6:
            if dev <= -U_EDGE and pos_u < LIMITS[U] - U_Q:
                q = min(U_Q, LIMITS[U] - pos_u)
                if q > 0:
                    orders[U].append(Order(U, ba_u, q))
            if dev >= U_EDGE and pos_u > 0:
                q = min(U_Q, pos_u)
                if q > 0:
                    orders[U].append(Order(U, bb_u, -q))

        if dev <= -U_EDGE:
            for sym in (VEV_5200, VEV_5300):
                d = state.order_depths.get(sym)
                if d is None:
                    continue
                bb, ba = _bb_ba(d)
                if bb is None or ba is None or ba - bb > MAX_VEV_SPREAD:
                    continue
                pos = int(state.position.get(sym, 0))
                if pos < LIMITS[sym] - VEV_FLOW_Q:
                    q = min(VEV_FLOW_Q, LIMITS[sym] - pos)
                    if q > 0:
                        orders[sym].append(Order(sym, ba, q))

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
