"""
v25: **vouchers_final_strategy** thesis only (see round3work/vouchers_final_strategy/STRATEGY.txt
and ORIGINAL_DISCORD_QUOTES.txt). No hydrogel, no Black–Scholes / gamma–scalping path.

- **Sonic / inclineGod:** VEV “flow” and small edges are only trustworthy when
  **VEV_5200** and **VEV_5300** **both** have BBO spread **≤ 2** (informed bot hedges
  into a **tight surface**; spread correlation matters, not just mids).
- **When gate is OFF (either leg wide):** post **no orders** — different regime,
  execution noise dominates (t-stat decays in community description).
- **When gate is ON:** (1) optional directional layer from the 6-panel story: build/maintain
  **long VELVETFRUIT_EXTRACT** toward a target; (2) market-make **only** **VEV_5200** and
  **VEV_5300** at BBO+edge so we trade **on** the tight legs Sonic names.

TTE / historical day: positions only; no IV engine. Tape day from __BT_TAPE_DAY__ for logging.
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

# Sonic recipe: both legs at or below 2 **at the same time**
SPREAD_TH = 2

# Extract: lean long when gate is on (STRATEGY “toy” long extract / long delta; sized conservatively)
EXTRACT_TARGET = 100
EXTRACT_BAND = 4
EXTRACT_MAX_CHILD = 22

# Two-leg surface only
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

        if bb_e is not None and ba_e is not None:
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
