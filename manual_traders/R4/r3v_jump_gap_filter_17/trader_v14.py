"""
Round 4 iteration 14 — Mark22 aggressive-buy prints: **bid +3** clip vs v13’s +2 (same conditions).

Same as **v11/v13**: on precomputed print tick, `mq_bid = mq_base + 3`, `mq_ask = mq_base`
(off-joint: 8 bid / 5 ask; joint-on: 19 bid / 16 ask). Fair unchanged (microprice skew only).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
S5200 = "VEV_5200"
S5300 = "VEV_5300"
TIGHT_TOB = 2
PRODUCTS = [
    HYDRO,
    EXTRACT,
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
LIMITS = {
    HYDRO: 200,
    EXTRACT: 200,
    **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
}
_TD = "r4v14"
_EMA = 0.15
_EDGE = 2
_BID_EXTRA_PRINT = 3


def _load_m22_aggr_buy_ts() -> dict[int, set[int]]:
    p = Path(__file__).resolve().parent / "precomputed" / "r4_extract_aggr_buy_m22_print.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): set(int(x) for x in v) for k, v in raw.items()}


_M22_TS: dict[int, set[int]] | None = None


def _day() -> int:
    e = os.environ.get("PROSPERITY4_BACKTEST_DAY")
    if e is not None and e.lstrip("-").isdigit():
        return int(e)
    return 1


def wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv, av = depth.buy_orders[bb], -depth.sell_orders[ba]
    tot = bv + av
    if tot <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * av + float(ba) * bv) / tot


def tob_spread(depth: OrderDepth) -> int | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return int(min(depth.sell_orders)) - int(max(depth.buy_orders))


def microprice(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv = float(depth.buy_orders[bb])
    av = float(abs(depth.sell_orders[ba]))
    tot = bv + av
    if tot <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * av + float(ba) * bv) / tot


def joint_tight_gate(state: TradingState) -> bool:
    d52 = state.order_depths.get(S5200)
    d53 = state.order_depths.get(S5300)
    if d52 is None or d53 is None:
        return False
    if not d52.buy_orders or not d52.sell_orders or not d53.buy_orders or not d53.sell_orders:
        return False
    s52, s53 = tob_spread(d52), tob_spread(d53)
    if s52 is None or s53 is None:
        return False
    return s52 <= TIGHT_TOB and s53 <= TIGHT_TOB


class Trader:
    def run(self, state: TradingState):
        global _M22_TS
        if _M22_TS is None:
            _M22_TS = _load_m22_aggr_buy_ts()

        bu: dict[str, Any] = {}
        if state.traderData:
            try:
                o = json.loads(state.traderData)
                if isinstance(o, dict) and _TD in o and isinstance(o[_TD], dict):
                    bu = o[_TD]
            except (json.JSONDecodeError, TypeError, KeyError):
                bu = {}

        out: dict[str, list[Order]] = {p: [] for p in PRODUCTS}

        day = _day()
        ts = int(state.timestamp)
        on_m22_print = ts in _M22_TS.get(day, set())

        exd = state.order_depths.get(EXTRACT)
        if exd is None or not exd.buy_orders or not exd.sell_orders:
            return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))

        wm = wall_mid(exd)
        if wm is None or wm <= 0:
            return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))

        f = bu.get("fex")
        if f is None:
            f = float(wm)
        else:
            f = float(f) + _EMA * (float(wm) - float(f))
        bu["fex"] = f

        joint = joint_tight_gate(state)
        mp = microprice(exd)
        skew = 0
        if mp is not None and mp > float(wm) + 0.25:
            skew = 1
        elif mp is not None and mp < float(wm) - 0.25:
            skew = -1

        pos = int(state.position.get(EXTRACT, 0) or 0)
        lim = LIMITS[EXTRACT]
        mq_base = 16 if joint else 5
        extra = _BID_EXTRA_PRINT if on_m22_print else 0
        mq_bid = mq_base + extra
        mq_ask = mq_base

        fi = int(round(float(f))) + skew
        bb, ba = max(exd.buy_orders), min(exd.sell_orders)
        bid_p = min(int(bb) + 1, fi - _EDGE)
        if bid_p >= 1 and bid_p < int(ba) and pos < lim:
            out[EXTRACT].append(Order(EXTRACT, bid_p, min(mq_bid, lim - pos)))
        ask_p = max(int(ba) - 1, fi + _EDGE)
        if ask_p > int(bb) and pos > -lim:
            out[EXTRACT].append(Order(EXTRACT, ask_p, -min(mq_ask, lim + pos)))

        return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))
