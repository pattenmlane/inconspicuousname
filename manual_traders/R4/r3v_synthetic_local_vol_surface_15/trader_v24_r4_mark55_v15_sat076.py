"""v15 band; SAT_GAIN 0.76 (stronger shrink at high roll)."""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
TH = 2
POS_LIMIT = 200
CLIP_MAX = 12
CLIP_MIN = 4
TRIGGER_BUYER = "Mark 55"
ROLL_WIN = 400
ROLL_LO = 0.20
ROLL_HI = 0.40
SAT_GAIN = 0.76


def _spread(depth):
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders.keys())
    ba = min(abs(p) for p in depth.sell_orders.keys())
    if ba <= bb:
        return None
    return int(ba - bb)


def _joint_tight(depths):
    s1 = _spread(depths.get(V5200))
    s2 = _spread(depths.get(V5300))
    if s1 is None or s2 is None:
        return False
    return s1 <= TH and s2 <= TH


def _bb_ba(depth):
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None, None
    bb = max(depth.buy_orders.keys())
    ba = min(abs(p) for p in depth.sell_orders.keys())
    if ba <= bb:
        return None, None
    return int(bb), int(ba)


def _clip_for_roll(roll: float) -> int:
    u = (roll - ROLL_LO) / max(ROLL_HI - ROLL_LO, 1e-9)
    u = max(0.0, min(1.0, u))
    mult = 1.0 - SAT_GAIN * u
    c = int(CLIP_MAX * mult + 0.5)
    return max(CLIP_MIN, min(CLIP_MAX, c))


class Trader:
    def run(self, state):
        depths = getattr(state, "order_depths", {}) or {}
        raw = getattr(state, "traderData", None) or ""
        try:
            store = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            store = {}
        if not isinstance(store, dict):
            store = {}

        day = int(getattr(state, "backtest_day", -1))
        if store.get("_roll_day") is not None and int(store["_roll_day"]) != day:
            store["tight_hist"] = []
        store["_roll_day"] = day

        tight_now = _joint_tight(depths)
        hist = store.get("tight_hist", [])
        if not isinstance(hist, list):
            hist = []
        hist.append(1 if tight_now else 0)
        if len(hist) > ROLL_WIN:
            hist = hist[-ROLL_WIN:]
        store["tight_hist"] = hist
        roll = sum(hist) / float(len(hist)) if hist else 0.0
        store["roll_tight_rate"] = roll

        if not tight_now:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        clip = _clip_for_roll(roll)
        store["last_clip"] = clip

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

        if not fired or pos >= POS_LIMIT - clip:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        q = min(clip, POS_LIMIT - pos)
        return {U: [Order(U, int(ba) + 1, q)]}, 0, json.dumps(store, separators=(",", ":"))
