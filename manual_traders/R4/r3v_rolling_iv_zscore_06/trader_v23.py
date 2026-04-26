"""
Round 4 Phase 2 — **Mark 67** counterparty-conditioned long on **VEV_5300** (Phase 1 edge).

Precomputed triggers: `outputs/r4_v23_signals.json` from `r4_phase2_analysis.py`
(Mark 67 aggressive buy on VELVETFRUIT_EXTRACT). Timestamps are on the **merged**
backtest timeline (same offset scheme as `ResultMerger`: cum offset per tape day).

When `state.timestamp` is within **W** ticks after any trigger, lift **VEV_5300** ask
(small clip) if spread is not too wide. When **no** trigger in window and we hold
VEV_5300, flatten to bid (Phase 2 execution / adverse-selection hygiene).

Hydrogel not traded. Limits per round4description.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

from datamodel import Order, OrderDepth, TradingState

HYDRO = "HYDROGEL_PACK"
UNDER = "VELVETFRUIT_EXTRACT"
V5300 = "VEV_5300"

Q = 6
MAX_SPREAD_V = 6
LIMITS = {
    HYDRO: 200,
    UNDER: 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    V5300: 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

_SIG_PATH = Path(__file__).resolve().parent / "outputs" / "r4_v23_signals.json"
_TRIGGERS: list[int] | None = None
_WINDOW: int = 50_000


def _load_signals() -> tuple[list[int], int]:
    global _TRIGGERS, _WINDOW
    if _TRIGGERS is not None:
        return _TRIGGERS, _WINDOW
    if not _SIG_PATH.is_file():
        _TRIGGERS = []
        return _TRIGGERS, _WINDOW
    obj = json.loads(_SIG_PATH.read_text())
    _TRIGGERS = sorted(int(x) for x in obj.get("mark67_extract_buy_aggr_abs_ts", []))
    _WINDOW = int(obj.get("window_ts", _WINDOW))
    return _TRIGGERS, _WINDOW


def _touch(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


def _active(ts: int, triggers: list[int], w: int) -> bool:
    if not triggers:
        return False
    lo = ts - w
    i = bisect.bisect_right(triggers, ts)
    j = bisect.bisect_left(triggers, lo)
    return j < i


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}

        triggers, w = _load_signals()
        ts = int(state.timestamp)
        on = _active(ts, triggers, w)

        result: dict[str, list[Order]] = {p: [] for p in LIMITS}
        pos = state.position
        d53 = state.order_depths.get(V5300)
        if d53 is None:
            return result, 0, json.dumps(td)

        vb, va = _touch(d53)
        if vb is None or va is None:
            return result, 0, json.dumps(td)
        sp = va - vb
        q = int(pos.get(V5300, 0))
        orders: list[Order] = []

        if on and sp <= MAX_SPREAD_V and q < LIMITS[V5300]:
            avail = abs(d53.sell_orders.get(va, 0))
            dq = min(Q, avail, LIMITS[V5300] - q)
            if dq > 0:
                orders.append(Order(V5300, va, dq))
        elif (not on) and q > 0:
            bb = max(d53.buy_orders)
            avail = d53.buy_orders.get(bb, 0)
            dq = min(q, avail, 30)
            if dq > 0:
                orders.append(Order(V5300, bb, -dq))

        result[V5300] = orders
        td["_sig_on"] = 1.0 if on else 0.0
        td["_ts"] = float(ts)
        return result, 0, json.dumps(td)
