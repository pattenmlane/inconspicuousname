"""
Round 4 follow-up — **Mark 22** as passive seller on **aggressive buys** (Phase 1 edge).

Precomputed triggers: `outputs/r4_v25_signals.json` from `r4_phase4_m22_joint_gate.py`
(`mark22_buyaggr_passive_seller_merged_ts`, merged backtest timeline).

Tape (R4 days 1–3, same-symbol forward mid K=5): n=109, mean **~2.07** ticks, **~85%**
positive (see `outputs/r4_p4_mark22_buyaggr_fwd5_joint_gate.txt`). Joint 5200+5300
tight gate **weakens** this aggregate on the full 109-print mix; this trader still
applies the **Sonic execution gate** (both spreads ≤2) when lifting **extract** for
realism / comparability with `trader_v24.py`.

When `state.timestamp` is in the post-trigger window, lift **VELVETFRUIT_EXTRACT** ask
(small clip) if extract spread is modest. Outside the window, flatten extract to bid.

Hydrogel not traded. Limits per round4description.
"""

from __future__ import annotations

import json
from pathlib import Path

from datamodel import Order, OrderDepth, TradingState

from merged_ts_util import window_active

HYDRO = "HYDROGEL_PACK"
UNDER = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"

Q = 8
MAX_SPREAD_U = 8
TH_JOINT = 2
LIMITS = {
    HYDRO: 200,
    UNDER: 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    V5200: 300,
    V5300: 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

_SIG_PATH = Path(__file__).resolve().parent / "outputs" / "r4_v25_signals.json"
_SIG_PACK: tuple[list[int], int, dict[int, int]] | None = None
_WINDOW: int = 50_000


def _load_signals() -> tuple[list[int], int, dict[int, int]]:
    global _SIG_PACK, _WINDOW
    if _SIG_PACK is not None:
        return _SIG_PACK
    if not _SIG_PATH.is_file():
        _SIG_PACK = ([], _WINDOW, {})
        return _SIG_PACK
    obj = json.loads(_SIG_PATH.read_text())
    tr = sorted(int(x) for x in obj.get("mark22_buyaggr_passive_seller_merged_ts", []))
    _WINDOW = int(obj.get("window_ts", _WINDOW))
    cum = {int(k): int(v) for k, v in obj.get("day_cum_offset", {}).items()}
    _SIG_PACK = (tr, _WINDOW, cum)
    return _SIG_PACK


def _touch(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


def _spread(depth: OrderDepth) -> int | None:
    vb, va = _touch(depth)
    if vb is None or va is None:
        return None
    return int(va - vb)


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}

        triggers, w, cum = _load_signals()
        ts = int(state.timestamp)
        on = window_active(state, ts, triggers, w, cum)

        result: dict[str, list[Order]] = {p: [] for p in LIMITS}
        pos = state.position
        du = state.order_depths.get(UNDER)
        d52 = state.order_depths.get(V5200)
        d53 = state.order_depths.get(V5300)
        if du is None or d52 is None or d53 is None:
            return result, 0, json.dumps(td)

        ub, ua = _touch(du)
        if ub is None or ua is None:
            return result, 0, json.dumps(td)
        sp_u = int(ua - ub)
        sp52 = _spread(d52)
        sp53 = _spread(d53)
        if sp52 is None or sp53 is None:
            return result, 0, json.dumps(td)

        joint_tight = sp52 <= TH_JOINT and sp53 <= TH_JOINT
        q = int(pos.get(UNDER, 0))
        orders: list[Order] = []

        if on and joint_tight and sp_u <= MAX_SPREAD_U and q < LIMITS[UNDER]:
            avail = abs(du.sell_orders.get(ua, 0))
            dq = min(Q, avail, LIMITS[UNDER] - q)
            if dq > 0:
                orders.append(Order(UNDER, ua, dq))
        elif (not on) and q > 0:
            bb = max(du.buy_orders)
            avail = du.buy_orders.get(bb, 0)
            dq = min(q, avail, 40)
            if dq > 0:
                orders.append(Order(UNDER, bb, -dq))

        result[UNDER] = orders
        td["_m22_on"] = 1.0 if on else 0.0
        td["_joint_tight"] = 1.0 if joint_tight else 0.0
        td["_ts"] = float(ts)
        return result, 0, json.dumps(td)
