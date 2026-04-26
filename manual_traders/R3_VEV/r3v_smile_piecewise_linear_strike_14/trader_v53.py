"""
v52 grid: larger size + looser EMA gate (tune for activity vs spread cost).

Same joint 5200+5300 <=2 gate and extract/VEV logic as v52; see STRATEGY.txt.
Products: VELVETFRUIT_EXTRACT + VEV_* only.
"""
from __future__ import annotations

import json

from datamodel import Order, TradingState

from _r3v_smile_core import book_walls, parse_td, synthetic_walls

U = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
VEV_TIGHT = (S5200, S5300)
VEV_FLAT = (S5200, S5300, "VEV_5000", "VEV_5100")

LIM_U = 200
LIM_V = 300

TH_SPREAD = 2
MOMENTUM_LONG = 0.25
MOMENTUM_SHORT = 0.25
EMA_S_WINDOW = 25
Q_EXTRACT = 25
Q_VEV = 4

FLAT_EXTRACT = 25
FLAT_VEV = 15

_TD_KEY = "vfinal_joint_gate_v2"


def _l1_fix(depth) -> tuple[int, int, int, float] | None:
    """(best_bid, best_ask, spread, wall_mid) or None if missing."""
    import math

    if depth is None:
        return None
    bw, aw, bb, ba, _wm0 = book_walls(depth)
    if bb is None or ba is None:
        return None
    _a, _b, wm, bb2, ba2 = synthetic_walls(bw, aw, bb, ba)
    if bb2 is None or ba2 is None:
        return None
    bb2i, ba2i = int(bb2), int(ba2)
    sp = ba2i - bb2i
    if wm is not None and math.isfinite(float(wm)):
        mid = float(wm)
    else:
        mid = 0.5 * (float(bb2i) + float(ba2i))
    if not math.isfinite(mid):
        return None
    return bb2i, ba2i, sp, mid


def _ema(store: dict[str, float], k: str, w: int, x: float) -> float:
    a = 2.0 / (w + 1.0)
    old = float(store.get(k, 0.0))
    n = a * x + (1.0 - a) * old
    store[k] = n
    return n


def _joint_tight(d52, d53) -> bool:
    t52 = _l1_fix(d52)
    t53 = _l1_fix(d53)
    if t52 is None or t53 is None:
        return False
    return t52[2] <= TH_SPREAD and t53[2] <= TH_SPREAD


class Trader:
    def run(self, state: TradingState):
        import math

        store = parse_td(getattr(state, "traderData", None))
        if not isinstance(store, dict):
            store = {}
        st: dict = store.get(_TD_KEY) if isinstance(store.get(_TD_KEY), dict) else {}
        if not isinstance(st, dict):
            st = {}
        st = {str(k): float(v) for k, v in st.items() if isinstance(v, (int, float))}

        depths = getattr(state, "order_depths", None) or {}
        pos = getattr(state, "position", None) or {}

        if U not in depths:
            store[_TD_KEY] = st
            return {}, 0, json.dumps(store, separators=(",", ":"))

        d52 = depths.get(S5200)
        d53 = depths.get(S5300)
        tight = _joint_tight(d52, d53)
        ut = _l1_fix(depths[U])
        if ut is None:
            store[_TD_KEY] = st
            return {}, 0, json.dumps(store, separators=(",", ":"))
        ubb, uba, _usp, umid = ut

        prev = st.get("S_prev")
        dmid = 0.0
        if prev is not None and math.isfinite(float(prev)):
            dmid = float(umid) - float(prev)
        st["S_prev"] = float(umid)
        ema_m = _ema(st, "dS", EMA_S_WINDOW, dmid)

        orders: dict[str, list[Order]] = {}
        p_u = int(pos.get(U, 0))

        def add(sym: str, ol: list[Order]) -> None:
            if ol:
                orders[sym] = ol

        if tight:
            if ema_m > MOMENTUM_LONG and p_u < LIM_U - Q_EXTRACT:
                add(U, [Order(U, uba, min(Q_EXTRACT, LIM_U - p_u))])
            elif ema_m < -MOMENTUM_SHORT and p_u > -LIM_U + Q_EXTRACT:
                add(U, [Order(U, ubb, -min(Q_EXTRACT, LIM_U + p_u))])
            for sym in VEV_TIGHT:
                d = depths.get(sym)
                t = _l1_fix(d)
                if t is None:
                    continue
                bb, ba, sp, _m = t
                if sp > TH_SPREAD:
                    continue
                pv = int(pos.get(sym, 0))
                if ema_m > MOMENTUM_LONG and pv < LIM_V - Q_VEV:
                    add(sym, [Order(sym, ba, min(Q_VEV, LIM_V - pv))])
                elif ema_m < -MOMENTUM_SHORT and pv > -LIM_V + Q_VEV:
                    add(sym, [Order(sym, bb, -min(Q_VEV, LIM_V + pv))])
        else:
            if p_u > 0:
                add(U, [Order(U, ubb, -min(FLAT_EXTRACT, p_u))])
            elif p_u < 0:
                add(U, [Order(U, uba, min(FLAT_EXTRACT, -p_u))])
            for sym in VEV_FLAT:
                d = depths.get(sym)
                t = _l1_fix(d)
                if t is None:
                    continue
                bb, ba, _sp, _m = t
                pv = int(pos.get(sym, 0))
                if pv > 0:
                    add(sym, [Order(sym, bb, -min(FLAT_VEV, pv))])
                elif pv < 0:
                    add(sym, [Order(sym, ba, min(FLAT_VEV, -pv))])

        store[_TD_KEY] = st
        return orders, 0, json.dumps(store, separators=(",", ":"))
