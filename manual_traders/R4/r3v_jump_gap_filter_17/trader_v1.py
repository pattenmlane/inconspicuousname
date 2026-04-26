"""
Round 4 iteration 1 — **Phase 2 microstructure + Sonic joint gate** (live book).

Tape Phase 2 showed near-coinstant extract Δ correlates with VEV_5200/5300 Δ (leadlag CSV);
Sonic-style **joint tight** = both VEV_5200 and VEV_5300 TOB spread ≤ 2.

Strategy (sim-safe; no counterparty in `TradingState` during run):
- **Joint gate ON:** larger extract clip, slightly skew quotes when extract **microprice > mid**
  (imbalance toward bid → lean long in mid space).
- **Joint gate OFF:** small extract-only clip (risk-off surface).

Counterparty-conditioned rules remain **tape-validated** in `outputs/phase2/`; extend runner
to expose `market_trades` if you need Mark IDs live.
"""
from __future__ import annotations

import json
import math
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
_TD = "r4v1"
_EMA = 0.15


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
        bu: dict[str, Any] = {}
        if state.traderData:
            try:
                o = json.loads(state.traderData)
                if isinstance(o, dict) and _TD in o and isinstance(o[_TD], dict):
                    bu = o[_TD]
            except (json.JSONDecodeError, TypeError, KeyError):
                bu = {}

        out: dict[str, list[Order]] = {p: [] for p in PRODUCTS}

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
        mq = 16 if joint else 5
        edge = 2
        fi = int(round(float(f))) + skew
        bb, ba = max(exd.buy_orders), min(exd.sell_orders)
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < lim:
            out[EXTRACT].append(Order(EXTRACT, bid_p, min(mq, lim - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -lim:
            out[EXTRACT].append(Order(EXTRACT, ask_p, -min(mq, lim + pos)))

        return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))
