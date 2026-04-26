"""
Round 4 — Mark 55 aggressive-buy extract under **Sonic joint gate** + **rarity filter**.

Tape analysis (`_r4_analyze_rolling_tight_prevalence.py`): on Round 4 days 1–3, rolling
400-tick fraction of (s5200≤2 & s5300≤2) is ~0.14 / ~0.17 / **~0.45** mean by day — day 3
is a **high gate-on prevalence** regime. v4’s loss concentrated there.

**Rule:** maintain last **ROLL_WIN** observations of whether the joint gate was tight at each tick.
If `sum(last) / len(last) > ROLL_MAX`, **do not** open new risk (skip Mark55 follow) even if
this tick is tight — avoids “stuck tight” / crowded periods. Same execution as v4 when allowed.

Constants tuned from `outputs_r4_phase3/r4_p6_rolling_tight_prevalence_by_day.csv` (day1–2 p90
~0.29–0.39; day3 mean ~0.45).
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState, Trade
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState, Trade

U = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
TH = 2
POS_LIMIT = 200
CLIP = 12
TRIGGER_BUYER = "Mark 55"
ROLL_WIN = 400
ROLL_MAX = 0.32


def _spread(depth: OrderDepth | None) -> int | None:
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders.keys())
    ba = min(abs(p) for p in depth.sell_orders.keys())
    if ba <= bb:
        return None
    return int(ba - bb)


def _joint_tight(depths: dict) -> bool:
    s1 = _spread(depths.get(V5200))
    s2 = _spread(depths.get(V5300))
    if s1 is None or s2 is None:
        return False
    return s1 <= TH and s2 <= TH


def _bb_ba(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None, None
    bb = max(depth.buy_orders.keys())
    ba = min(abs(p) for p in depth.sell_orders.keys())
    if ba <= bb:
        return None, None
    return int(bb), int(ba)


class Trader:
    def run(self, state: TradingState):
        depths = getattr(state, "order_depths", {}) or {}
        raw = getattr(state, "traderData", None) or ""
        try:
            store: dict[str, Any] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            store = {}
        if not isinstance(store, dict):
            store = {}

        day = int(getattr(state, "backtest_day", -1))
        prev_day = store.get("_roll_day", None)
        if prev_day is not None and int(prev_day) != day:
            store["tight_hist"] = []
        store["_roll_day"] = day

        tight_now = _joint_tight(depths)
        hist: list[int] = store.get("tight_hist", [])
        if not isinstance(hist, list):
            hist = []
        hist.append(1 if tight_now else 0)
        if len(hist) > ROLL_WIN:
            hist = hist[-ROLL_WIN:]
        store["tight_hist"] = hist

        roll = sum(hist) / float(len(hist)) if hist else 0.0
        store["roll_tight_rate"] = roll

        if not tight_now or roll > ROLL_MAX:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        pos = int((getattr(state, "position", {}) or {}).get(U, 0))
        mkt = getattr(state, "market_trades", None) or {}
        if not isinstance(mkt, dict):
            mkt = {}

        depth_u = depths.get(U)
        bb, ba = _bb_ba(depth_u)
        if bb is None or ba is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        fired = False
        for tr in mkt.get(U, []):
            if getattr(tr, "buyer", None) != TRIGGER_BUYER:
                continue
            if int(getattr(tr, "price", 0)) >= ba:
                fired = True
                break

        if not fired or pos >= POS_LIMIT - CLIP:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        q = min(CLIP, POS_LIMIT - pos)
        return {U: [Order(U, int(ba) + 1, q)]}, 0, json.dumps(store, separators=(",", ":"))
