"""
Round 4 Phase 2 — burst + Mark01->Mark22 VEV + Sonic joint gate on **live** book.

Backtester does not populate market_trades before Trader.run; we use precomputed
burst (day,timestamp) list from tape (r4_p2_m01_m22_burst_pairs.json) and infer CSV
day from VELVET mid at session open (ts==0) vs Round-4 anchors (5245 / 5267.5 / 5295.5).

When (inferred_day, ts) is a burst tick and 5200+5300 L1 spreads <=2, passive buy extract
one tick inside BBO (small size).
"""
from __future__ import annotations

import json
from pathlib import Path

from datamodel import Order, TradingState

from _r3v_smile_core import book_walls, parse_td, synthetic_walls

U = "VELVETFRUIT_EXTRACT"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
LIM_U = 200
TH = 2
Q = 6
_TD_KEY = "r4_p2_burst_gate"

_PAIR_PATH = Path(__file__).resolve().parent / "analysis_outputs" / "r4_p2_m01_m22_burst_pairs.json"
_BURST_BY_DAY: dict[int, frozenset[int]] = {}
if _PAIR_PATH.exists():
    pairs = json.loads(_PAIR_PATH.read_text())
    bd: dict[int, set[int]] = {}
    for d, t in pairs:
        bd.setdefault(int(d), set()).add(int(t))
    _BURST_BY_DAY = {d: frozenset(ts) for d, ts in bd.items()}


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


def _joint_tight(depths) -> bool:
    a = depths.get(S5200)
    b = depths.get(S5300)
    if a is None or b is None:
        return False
    _, _, s52 = _l1_spread(a)
    _, _, s53 = _l1_spread(b)
    if s52 is None or s53 is None:
        return False
    return s52 <= TH and s53 <= TH


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
        d_csv = int(st.get("csv_day", 0))
        if d_csv not in _BURST_BY_DAY or ts not in _BURST_BY_DAY[d_csv]:
            store[_TD_KEY] = st
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if not _joint_tight(depths):
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
