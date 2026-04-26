"""
Round 4 — **v5** + **inventory-aware surface MM** (Phase 2 / suggested direction).

Same as **trader_v5** (Sonic joint gate + Mark 67 aggressive extract lift +
VEV_5200/5300 MM at BBO+1, `mm_tick` in `traderData`).

**Skew:** when net **long** a surface leg (`pos > 0`), post a **smaller buy** clip
and keep sell clip normal; when net **short** (`pos < 0`), post a **smaller sell**
clip and keep buy clip normal. When flat, symmetric **OPTION_CLIP**.

Rationale: day-3 losses in v3/v5 surface MM are dominated by adverse inventory;
reducing the side that **adds** to the wrong-way position may cut drag under
`worse` fills without turning off the gate thesis.
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
MAX_EX_POS = 80
CLIP_EX = 12
MM_EDGE = 1
OPTION_CLIP = 10
SKEWED_CLIP = 6
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


def clips_for_position(pos: int) -> tuple[int, int]:
    """(buy_clip, sell_clip) max 1 contract."""
    base = max(1, OPTION_CLIP)
    skew = max(1, SKEWED_CLIP)
    if pos > 0:
        return min(skew, base), base
    if pos < 0:
        return base, min(skew, base)
    return base, base


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        conversions = 0
        try:
            store: dict[str, Any] = json.loads(state.traderData) if (state.traderData or "").strip() else {}
        except (json.JSONDecodeError, TypeError):
            store = {}
        obs = getattr(state.observations, "plainValueObservations", None) or {}

        if not joint_tight_gate(state):
            store["sonic_tight"] = False
            if "__BT_TAPE_DAY__" in obs:
                store["tape_day"] = int(obs["__BT_TAPE_DAY__"])
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
                c_buy, c_sell = clips_for_position(pos)
                if room_buy > 0:
                    result[sym].append(Order(sym, bid_px, min(c_buy, room_buy)))
                if room_sell > 0 and pos > 0:
                    result[sym].append(Order(sym, sell_px, -min(c_sell, pos, room_sell)))

        if "__BT_TAPE_DAY__" in obs:
            store["tape_day"] = int(obs["__BT_TAPE_DAY__"])
        return result, conversions, json.dumps(store)
