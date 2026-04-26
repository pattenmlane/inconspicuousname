"""**Q=5** + **EOD_PAD=780** (v50 uses 800; test EOD×clip interaction)."""

from __future__ import annotations

import json
from pathlib import Path

from datamodel import Order, OrderDepth, TradingState

from merged_ts_util import window_active

HYDRO = "HYDROGEL_PACK"
UNDER = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"

Q = 5
MAX_SPREAD_V = 6
TH_JOINT = 2
DAY_END = 999_900
EOD_PAD = 780
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

_SIG_PATH = Path(__file__).resolve().parent / "outputs" / "r4_v26_signals.json"
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
    tr = sorted(int(x) for x in obj.get("mark67_extract_buy_aggr_filtered_merged_ts", []))
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
        day = int(getattr(state, "day_num", -1))

        result: dict[str, list[Order]] = {p: [] for p in LIMITS}
        pos = state.position
        d53 = state.order_depths.get(V5300)
        d52 = state.order_depths.get(V5200)
        if d53 is None or d52 is None:
            return result, 0, json.dumps(td)

        vb, va = _touch(d53)
        if vb is None or va is None:
            return result, 0, json.dumps(td)
        sp53 = int(va - vb)
        sp52 = _spread(d52)
        if sp52 is None:
            return result, 0, json.dumps(td)

        joint_tight = sp52 <= TH_JOINT and sp53 <= TH_JOINT
        q = int(pos.get(V5300, 0))
        orders: list[Order] = []

        eod_flat = day in (1, 2) and ts >= DAY_END - EOD_PAD and q > 0

        if eod_flat:
            bb = max(d53.buy_orders)
            avail = d53.buy_orders.get(bb, 0)
            dq = min(q, avail, 30)
            if dq > 0:
                orders.append(Order(V5300, bb, -dq))
        elif on and joint_tight and sp53 <= MAX_SPREAD_V and q < LIMITS[V5300]:
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
        td["_joint_tight"] = 1.0 if joint_tight else 0.0
        td["_eod"] = 1.0 if eod_flat else 0.0
        td["_ts"] = float(ts)
        return result, 0, json.dumps(td)
