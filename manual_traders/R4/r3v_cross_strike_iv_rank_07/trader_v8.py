"""
Round 4 iteration 8 — Phase 1 two-hop chain (Mark 55→Mark 01→Mark 22) × Sonic gate (Phase 3 stack).

Offline: r4_p1_twohop_m55_m01_m22_summary.json — 535 events, pooled extract dm_ex k5/k20/k100 means
≈ -0.21 / -0.64 / -1.39 (t ≈ -2.39 / -3.44 / -3.44). Per-day: day1 k20 n.s.; day2 k5 positive; day3
strong negative at k5/k20 — chain is not uniformly stable by day.

Sim (stateful):
  Arm when (Sonic tight) AND tape shows Mark 55 buying VELVETFRUIT_EXTRACT from Mark 01 at/above ask.
  If armed and arm_ts < current ts <= arm_ts + WIN (5000) AND Sonic tight AND tape shows Mark 01
  selling any product to Mark 22 (seller Mark 22, buyer Mark 01), short extract at best bid (clip 20).
  Clear arm after a fire or when window expires. Cooldown 4 buckets between fires.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH = 2
WIN = 5000
CLIP = 20
EX_LIM = 200
COOLDOWN = 4
WARMUP = 5

_KEY_ARM = "arm_ts"
_KEY_LAST = "last_fire_v8"


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _sym(state: TradingState, prod: str) -> str | None:
    for s, lst in (getattr(state, "listings", {}) or {}).items():
        if getattr(lst, "product", None) == prod:
            return s
    return None


def _ba(d: OrderDepth | None) -> tuple[int | None, int | None]:
    if d is None:
        return None, None
    b, s = getattr(d, "buy_orders", None) or {}, getattr(d, "sell_orders", None) or {}
    if not b or not s:
        return None, None
    return max(b.keys()), min(s.keys())


def _sonic(depths: dict[str, OrderDepth], s520: str, s530: str) -> bool:
    b5, a5 = _ba(depths.get(s520))
    b3, a3 = _ba(depths.get(s530))
    if None in (b5, a5, b3, a3):
        return False
    return (a5 - b5) <= TH and (a3 - b3) <= TH


def _leg1_arm(state: TradingState, sym_ex: str, ask: int) -> bool:
    for tr in (getattr(state, "market_trades", None) or {}).get(sym_ex, []) or []:
        if (
            getattr(tr, "buyer", None) == "Mark 55"
            and getattr(tr, "seller", None) == "Mark 01"
            and int(getattr(tr, "price", 0)) >= int(ask)
        ):
            return True
    return False


def _leg2_fire(state: TradingState) -> bool:
    m = getattr(state, "market_trades", None) or {}
    for trlist in m.values():
        for tr in trlist or []:
            if getattr(tr, "buyer", None) == "Mark 01" and getattr(tr, "seller", None) == "Mark 22":
                return True
    return False


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        sym_ex = _sym(state, "VELVETFRUIT_EXTRACT")
        s520 = _sym(state, "VEV_5200")
        s530 = _sym(state, "VEV_5300")
        if not sym_ex or not s520 or not s530:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        tight = _sonic(depths, s520, s530)

        arm = td.get(_KEY_ARM)
        if isinstance(arm, int):
            if ts - arm > WIN:
                td.pop(_KEY_ARM, None)
                arm = None

        d = depths.get(sym_ex)
        bid, ask = _ba(d)
        if bid is None or ask is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        out: dict[str, list[Order]] = {}
        if tight and isinstance(arm, int) and ts > arm and _leg2_fire(state):
            bucket = ts // 100
            last = td.get(_KEY_LAST)
            if not (isinstance(last, int) and bucket - last < COOLDOWN):
                p = int(pos.get(sym_ex, 0))
                room = p + EX_LIM
                q = min(CLIP, room)
                if q > 0:
                    td[_KEY_LAST] = int(bucket)
                    td.pop(_KEY_ARM, None)
                    out[sym_ex] = [Order(sym_ex, int(bid), -q)]

        if tight and _leg1_arm(state, sym_ex, int(ask)):
            td[_KEY_ARM] = int(ts)

        return out, 0, json.dumps(td, separators=(",", ":"))
