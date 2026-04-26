"""
Round 4 — **Hybrid Mark 67 extract**: **v19** sizing when **Sonic joint gate on**;
**small** clip/cap when gate **off**.

- **Joint tight** (5200+5300 spread≤2): same Mark 67 follow as **v19**
  (**CLIP_EX=20**, **MAX_EX_POS=120**) + surface MM rules (**v8** day-3 surface off).
- **Gate off**: still follow Mark 67 aggressive extract (price≥ask) but only
  **CLIP_EX_LOOSE** lots per signal and cap **MAX_EX_POS_LOOSE** — probes whether
  **v26**’s loss comes from **size** on the ~87% of prints that are not joint-tight
  (`r4_mark67_aggr_extract_joint_gate_counts_by_day.csv`), without abandoning the
  gate for the main extract leg.
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
CLIP_EX_TIGHT = 20
MAX_EX_TIGHT = 120
CLIP_EX_LOOSE = 4
MAX_EX_LOOSE = 40
MM_EDGE = 1
OPTION_CLIP = 10
REQUOTE_EVERY = 2

SKIP_SURFACE_MM_TAPE_DAYS = frozenset({3})


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


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        conversions = 0
        try:
            store: dict[str, Any] = json.loads(state.traderData) if (state.traderData or "").strip() else {}
        except (json.JSONDecodeError, TypeError):
            store = {}
        obs = getattr(state.observations, "plainValueObservations", None) or {}

        tape_day = int(obs["__BT_TAPE_DAY__"]) if "__BT_TAPE_DAY__" in obs else 0
        store["tape_day"] = tape_day

        raw = obs.get("__BT_TAPE_TRADES_JSON__", "[]")
        try:
            tape = json.loads(raw) if isinstance(raw, str) else []
        except json.JSONDecodeError:
            tape = []

        tight = joint_tight_gate(state)
        clip_ex = CLIP_EX_TIGHT if tight else CLIP_EX_LOOSE
        max_ex = MAX_EX_TIGHT if tight else MAX_EX_LOOSE
        store["sonic_tight"] = tight
        if tight:
            store["m67_clip_mode"] = "tight"
        else:
            store["m67_clip_mode"] = "loose"

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
            if pos_ex >= max_ex:
                continue
            room = LIMITS[EXTRACT] - pos_ex
            q = min(clip_ex, room)
            if q > 0:
                result[EXTRACT].append(Order(EXTRACT, int(ba), int(q)))
                store["m67_follow"] = True
                break

        if not tight:
            return result, conversions, json.dumps(store)

        if tape_day in SKIP_SURFACE_MM_TAPE_DAYS:
            store["surface_mm_off"] = True
            return result, conversions, json.dumps(store)

        tick = state.timestamp // REQUOTE_EVERY
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
