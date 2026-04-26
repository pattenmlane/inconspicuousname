"""
Round 4 iteration 5 — diagnostic **touch takers** on VEV_5200/5300 when Sonic joint gate is on.

v3/v4 never got VEV PnL because passive quotes at bid+1 do not cross a 2-tick book, and the
`taker` branch required `ask <= fair - edge` with **wall_mid** fair inside [bid,ask] — almost
never true.

v5: when joint tight, each tick place **one** small buy at **best ask** and **one** small sell at
**best bid** on each gate leg (qty 2) — these **must** match `sell_orders` / `buy_orders` if depth
exists. Extract quoting unchanged from v1/v2. Expect noisy PnL; purpose is to prove **fill path**
then replace with economics.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
S5200, S5300 = "VEV_5200", "VEV_5300"
GATE = (S5200, S5300)
TIGHT_TOB = 2
PRODUCTS = [
    HYDRO,
    EXTRACT,
    *[f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)],
]
LIMITS = {
    HYDRO: 200,
    EXTRACT: 200,
    **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
}
_TD = "r4v5"
_EMA = 0.15
_TOUCH_Q = 2


def wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv, av = depth.buy_orders[bb], -depth.sell_orders[ba]
    tot = bv + av
    if tot <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * av + float(ba) * bv) / tot


def tob_spread(d: OrderDepth) -> int | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return int(min(d.sell_orders)) - int(max(d.buy_orders))


def joint_tight(state: TradingState) -> bool:
    a, b = state.order_depths.get(S5200), state.order_depths.get(S5300)
    if not a or not b or not a.buy_orders or not a.sell_orders or not b.buy_orders or not b.sell_orders:
        return False
    x, y = tob_spread(a), tob_spread(b)
    return x is not None and y is not None and x <= TIGHT_TOB and y <= TIGHT_TOB


def microprice(d: OrderDepth) -> float | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    bb, ba = max(d.buy_orders), min(d.sell_orders)
    bv, av = float(d.buy_orders[bb]), float(abs(d.sell_orders[ba]))
    t = bv + av
    if t <= 0:
        return 0.5 * (float(bb) + float(ba))
    return (float(bb) * av + float(ba) * bv) / t


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

        j = joint_tight(state)
        mp = microprice(exd)
        sk = 0
        if mp and mp > float(wm) + 0.25:
            sk = 1
        elif mp and mp < float(wm) - 0.25:
            sk = -1

        pos = int(state.position.get(EXTRACT, 0) or 0)
        lim = LIMITS[EXTRACT]
        mq = 14 if j else 5
        fi = int(round(float(f))) + sk
        bb, ba = max(exd.buy_orders), min(exd.sell_orders)
        bp = min(int(bb) + 1, fi - 2)
        if bp >= 1 and bp < int(ba) and pos < lim:
            out[EXTRACT].append(Order(EXTRACT, bp, min(mq, lim - pos)))
        ap = max(int(ba) - 1, fi + 2)
        if ap > int(bb) and pos > -lim:
            out[EXTRACT].append(Order(EXTRACT, ap, -min(mq, lim + pos)))

        if j:
            for sym in GATE:
                d = state.order_depths.get(sym)
                if d is None or not d.buy_orders or not d.sell_orders:
                    continue
                bbi, bai = int(max(d.buy_orders)), int(min(d.sell_orders))
                posv = int(state.position.get(sym, 0) or 0)
                limv = LIMITS[sym]
                if posv < limv:
                    out[sym].append(Order(sym, bai, min(_TOUCH_Q, limv - posv)))
                if posv > -limv:
                    out[sym].append(Order(sym, bbi, -min(_TOUCH_Q, limv + posv)))

        return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))
