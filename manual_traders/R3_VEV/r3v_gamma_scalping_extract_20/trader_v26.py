"""
v26: `v25` with **no long-extract layer on the last historical tape** (here: tape
day index **3** when four CSV days are present for Round 3).

Rationale: STRATEGY.txt warns mid-forward edge is a heuristic, not PnL; the
K=20 / tight-gate story was calibrated on earlier-day tapes in our pack. On the
shortest-horizon day, aggressively lifting extract at the ask to hit a long
**target** produced a large **negative** extract line in v25. **v26** still
enforces the joint spread gate and still **market-makes 5200+5300** on every
tight day; it only **skips the directional extract accumulation** on tape
day **≥ 3** (surface MM only in that regime).
"""
from __future__ import annotations

import json
from typing import Any

from datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
SURFACE = (VEV_5200, VEV_5300)

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_ALL = [f"VEV_{k}" for k in STRIKES]
PRODUCTS = [EXTRACT] + VEV_ALL

LIMITS: dict[str, int] = {EXTRACT: 200, **{v: 300 for v in VEV_ALL}}

SPREAD_TH = 2

EXTRACT_TARGET = 100
EXTRACT_BAND = 4
EXTRACT_MAX_CHILD = 22
# Last-tape day index (0-based) at which we stop extract targeting (v25 ablation)
SKIP_EXTRACT_TARGET_FROM_TAPE_DAY = 3

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
            tape_day = int(obs["__BT_TAPE_DAY__"])
        else:
            last_ts = int(store.get("last_ts", -1))
            tape_day = int(store.get("tape_day", 0))
            if state.timestamp == 0 and last_ts > 50_000:
                tape_day = min(tape_day + 1, 3)
            store["tape_day"] = tape_day
            store["last_ts"] = int(state.timestamp)
        store["bt_tape_day"] = tape_day

        if not joint_tight_gate(state):
            store["tight_gate"] = False
            return result, conversions, json.dumps(store)

        store["tight_gate"] = True

        d_ex = state.order_depths.get(EXTRACT)
        bb_e, ba_e = _best(d_ex)
        pos_ex = int(state.position.get(EXTRACT, 0))

        if tape_day < SKIP_EXTRACT_TARGET_FROM_TAPE_DAY and bb_e is not None and ba_e is not None:
            target = min(EXTRACT_TARGET, LIMITS[EXTRACT])
            if pos_ex < target - EXTRACT_BAND and pos_ex < LIMITS[EXTRACT]:
                room = LIMITS[EXTRACT] - pos_ex
                q = min(target - pos_ex, room, EXTRACT_MAX_CHILD)
                if q > 0:
                    result[EXTRACT].append(Order(EXTRACT, int(ba_e), int(q)))
            elif pos_ex > target + EXTRACT_BAND and pos_ex > -LIMITS[EXTRACT]:
                q = min(pos_ex - target, pos_ex + LIMITS[EXTRACT], EXTRACT_MAX_CHILD)
                if q > 0:
                    result[EXTRACT].append(Order(EXTRACT, int(bb_e), -int(q)))

        tick = state.timestamp // REQUOTE_EVERY
        if tick == int(store.get("mm_tick", -1)):
            return result, conversions, json.dumps(store)
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
