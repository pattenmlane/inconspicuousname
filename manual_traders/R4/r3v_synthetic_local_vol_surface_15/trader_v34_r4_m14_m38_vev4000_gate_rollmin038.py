"""
v31 (Mark14/M38 aggr_sell VEV_4000 + Sonic gate) + **rolling gate prevalence** filter.

Tape (`r4_p17_...csv` from `_r4_m14_m38_vev4000_roll_at_print.py`): on **day 3** (and **day 2**),
prints with **low** roll_tight_rate (WIN=400) have **worse** mean fwd20 than **high** roll
(e.g. day3: roll<=0.38 mean \u2248\u22121.28 vs roll>0.38 mean \u2248+0.99). Day 1 is milder.

**Rule:** only fire when joint gate is tight **and** `roll_tight_rate > ROLL_MIN` (buffer reset per
`backtest_day` like v7/v15). Default ROLL_MIN=0.38 (between day1\u20132 p90 ~0.29\u20130.39 and day3 mean ~0.47).
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

SYM = "VEV_4000"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
TH = 2
POS_LIM = 300
CLIP = 8
ROLL_WIN = 400
ROLL_MIN = 0.38


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
        if store.get("_roll_day") is not None and int(store["_roll_day"]) != day:
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

        if not tight_now or roll <= ROLL_MIN:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        pos = int((getattr(state, "position", {}) or {}).get(SYM, 0))
        mkt = getattr(state, "market_trades", None) or {}
        if not isinstance(mkt, dict):
            mkt = {}

        depth = depths.get(SYM)
        bb, ba = _bb_ba(depth)
        if bb is None or ba is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        fired = False
        for tr in mkt.get(SYM, []):
            if getattr(tr, "buyer", None) != "Mark 14":
                continue
            if getattr(tr, "seller", None) != "Mark 38":
                continue
            if int(getattr(tr, "price", 0)) > bb:
                continue
            fired = True
            break

        if not fired or pos >= POS_LIM - CLIP:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        q = min(CLIP, POS_LIM - pos)
        return {SYM: [Order(SYM, int(ba) + 1, q)]}, 0, json.dumps(store, separators=(",", ":"))
