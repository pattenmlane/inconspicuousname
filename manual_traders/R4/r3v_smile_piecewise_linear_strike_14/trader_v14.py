"""
Round 4 iteration 14 — Hybrid Sonic gate (asymmetric thresholds).

VEV_5200 L1 <= 2 and VEV_5300 L1 <= 3 — ablation between v2 (2,2) and v13 (3,3).
"""
from __future__ import annotations

import json

from datamodel import Order, TradingState

from _r3v_smile_core import book_walls, parse_td, synthetic_walls

U = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
LIM_U = 200
TH_52 = 2
TH_53 = 3
Q = 4
_TD_KEY = "r4_gate_hybrid_2_3"


def _mid_open(depth) -> float | None:
    import math

    if depth is None:
        return None
    bw, aw, bb, ba, _ = book_walls(depth)
    if bb is None or ba is None:
        return None
    _, _, wm, bb2, ba2 = synthetic_walls(bw, aw, bb, ba)
    if wm is not None and math.isfinite(float(wm)):
        return float(wm)
    if bb2 is not None and ba2 is not None:
        return 0.5 * (float(bb2) + float(ba2))
    return None


def _infer_csv_day(mid: float) -> int | None:
    if abs(mid - 5245.0) < 4.0:
        return 1
    if abs(mid - 5267.5) < 4.0:
        return 2
    if abs(mid - 5295.5) < 4.0:
        return 3
    return None


def _l1_spread(depth) -> tuple[int | None, int | None, int | None]:
    if depth is None:
        return None, None, None
    bw, aw, bb, ba, _ = book_walls(depth)
    if bb is None or ba is None:
        return None, None, None
    _, _, _, bb2, ba2 = synthetic_walls(bw, aw, bb, ba)
    if bb2 is None or ba2 is None:
        return int(bb), int(ba), int(ba) - int(bb)
    return int(bb2), int(ba2), int(ba2) - int(bb2)


def _joint_hybrid(depths) -> bool:
    a = depths.get(S5200)
    b = depths.get(S5300)
    if a is None or b is None:
        return False
    _, _, s52 = _l1_spread(a)
    _, _, s53 = _l1_spread(b)
    if s52 is None or s53 is None:
        return False
    return s52 <= TH_52 and s53 <= TH_53


def _passive_buy_px(bb: int, ba: int) -> int:
    if ba - bb >= 2:
        return min(bb + 1, ba - 1)
    return bb


class Trader:
    def run(self, state: TradingState):
        store = parse_td(getattr(state, "traderData", None))
        if not isinstance(store, dict):
            store = {}
        st = store.get(_TD_KEY)
        if not isinstance(st, dict):
            st = {}
        st = {str(k): float(v) for k, v in st.items() if isinstance(v, (int, float))}

        ts = int(getattr(state, "timestamp", 0))
        depths = getattr(state, "order_depths", None) or {}
        pos = getattr(state, "position", None) or {}

        if "csv_day" not in st and ts == 0 and U in depths:
            m0 = _mid_open(depths[U])
            if m0 is not None:
                cd = _infer_csv_day(m0)
                if cd is not None:
                    st["csv_day"] = float(cd)

        if not _joint_hybrid(depths):
            store[_TD_KEY] = st
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if U not in depths:
            store[_TD_KEY] = st
            return {}, 0, json.dumps(store, separators=(",", ":"))

        bb, ba, _ = _l1_spread(depths[U])
        if bb is None or ba is None:
            store[_TD_KEY] = st
            return {}, 0, json.dumps(store, separators=(",", ":"))

        p_u = int(pos.get(U, 0))
        if p_u >= LIM_U - Q:
            store[_TD_KEY] = st
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if int(st.get("last_act_ts", -1)) == ts:
            store[_TD_KEY] = st
            return {}, 0, json.dumps(store, separators=(",", ":"))

        st["last_act_ts"] = float(ts)
        store[_TD_KEY] = st
        px = _passive_buy_px(bb, ba)
        return {U: [Order(U, px, min(Q, LIM_U - p_u))]}, 0, json.dumps(store, separators=(",", ":"))
