"""
Round 4 — **v5** + **counterparty skip** on dominant surface print.

Offline table `r4_joint_tight_vev_print_top_edges_by_day.csv` (from
`r4_joint_tight_vev_print_counterparties.py`): among **joint-tight**
(5200+5300 spread≤2) timestamps, **Mark 01 → Mark 22** on **VEV_5200** or
**VEV_5300** is the top edge every tape day (e.g. day 3: **80**/111 prints).

When the Sonic gate is **on**, if **this tick's** `__BT_TAPE_TRADES_JSON__`
contains such a print on **5200 or 5300**, **skip** posting VEV surface quotes
for this requote interval (same cadence as v5). **Mark 67** extract lift is
unchanged.

This tests a **counterparty-conditioned** alternative to calendar **v8**
(day-3 surface off).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
SURFACE = (VEV_5200, VEV_5300)
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV = [f"VEV_{k}" for k in STRIKES]
PRODUCTS = [HYDRO, EXTRACT] + VEV

LIMITS = {HYDRO: 200, EXTRACT: 200, **{v: 300 for v in VEV}}

SPREAD_TH = 2
MARK67 = "Mark 67"
MARK01 = "Mark 01"
MARK22 = "Mark 22"
MAX_EX_POS = 80
CLIP_EX = 12
MM_EDGE = 1
OPTION_CLIP = 10
REQUOTE_EVERY = 2


def _best(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if not depth:
        return None, None
    bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return bb, ba


def _bbo_spread(depth: OrderDepth | None) -> int | None:
    bb, ba = _best(depth)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def joint_tight_gate(state: TradingState) -> bool:
    s0 = _bbo_spread(state.order_depths.get(VEV_5200))
    s1 = _bbo_spread(state.order_depths.get(VEV_5300))
    if s0 is None or s1 is None:
        return False
    return s0 <= SPREAD_TH and s1 <= SPREAD_TH


def mark01_to_22_surface_on_tape(tape: list[dict]) -> bool:
    for t in tape:
        if str(t.get("symbol", "")) not in SURFACE:
            continue
        if t.get("buyer") == MARK01 and t.get("seller") == MARK22:
            return True
    return False


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        conversions = 0
        try:
            store: dict[str, Any] = json.loads(state.traderData) if (state.traderData or "").strip() else {}
        except (json.JSONDecodeError, TypeError):
            store = {}
        obs = getattr(state.observations, "plainValueObservations", None) or {}

        if "__BT_TAPE_DAY__" in obs:
            store["tape_day"] = int(obs["__BT_TAPE_DAY__"])

        if not joint_tight_gate(state):
            store["sonic_tight"] = False
            return result, conversions, json.dumps(store)

        store["sonic_tight"] = True

        raw = obs.get("__BT_TAPE_TRADES_JSON__", "[]")
        try:
            tape = json.loads(raw) if isinstance(raw, str) else []
        except json.JSONDecodeError:
            tape = []

        for t in tape:
            if t.get("buyer") != MARK67 or t.get("symbol") != EXTRACT:
                continue
            pr = int(t.get("price", -1))
            d_ex = state.order_depths.get(EXTRACT)
            bb, ba = _best(d_ex)
            if bb is None or ba is None:
                continue
            if pr < ba:
                continue
            pos_ex = int(state.position.get(EXTRACT, 0))
            if pos_ex >= MAX_EX_POS:
                continue
            room = LIMITS[EXTRACT] - pos_ex
            q = min(CLIP_EX, room)
            if q > 0:
                result[EXTRACT].append(Order(EXTRACT, int(ba), int(q)))
                store["m67_follow"] = True
                break

        tick = state.timestamp // REQUOTE_EVERY
        if mark01_to_22_surface_on_tape(tape):
            store["skip_surface_mm_m01_m22"] = True
            store["mm_tick"] = tick
            return result, conversions, json.dumps(store)

        if tick != int(store.get("mm_tick", -1)):
            store["mm_tick"] = tick
            for sym in SURFACE:
                d = state.order_depths.get(sym)
                if not d:
                    continue
                bb, ba = _best(d)
                if bb is None or ba is None:
                    continue
                pos = int(state.position.get(sym, 0))
                room_buy = LIMITS[sym] - pos
                room_sell = LIMITS[sym] + pos
                bid_px = min(int(bb) + MM_EDGE, int(ba) - 1)
                sell_px = max(int(ba) - MM_EDGE, int(bb) + 1)
                c = OPTION_CLIP
                if room_buy > 0:
                    result[sym].append(Order(sym, bid_px, min(c, room_buy)))
                if room_sell > 0 and pos > 0:
                    result[sym].append(Order(sym, sell_px, -min(c, pos, room_sell)))

        return result, conversions, json.dumps(store)
