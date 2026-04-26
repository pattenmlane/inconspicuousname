"""
Round 4 iteration 7 — v6 **execution hygiene** on the same precomputed Mark01→Mark22 ∩ joint-tight set.

v6 placed a **sell at best bid on VEV_5300 on every tick** where the offline signal was true
(dense windows around prints → large short inventory and adverse sim fills). v7 keeps extract
logic identical to v6/v1 but only fades 5300 when:

- joint gate on and timestamp in precomputed signal set (same JSON as v6);
- **VEV_5300 TOB spread ≥ 2** (skip 1-tick books where repeated bid supply is worst);
- **cooldown** ≥ 500 time units since last 5300 fade order (one burst per ~5 ticks max);
- **qty 2** per eligible tick (was 6 in v6).

Hypothesis: tape short-horizon negative fwd is real but **sim PnL** needs throttling + spread room.
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
S5200, S5300 = "VEV_5200", "VEV_5300"
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
_TD = "r4v7"
_EMA = 0.15
_FADE_Q = 2
_MIN_S5300_SPREAD = 2
_COOLDOWN_TS = 500  # 5 ticks × 100 time units (same grid as preprocess WINDOW_TICKS)


def _load_signal_set() -> dict[int, set[int]]:
    p = Path(__file__).resolve().parent / "precomputed" / "r4_m0122_5300_window_joint_signal.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): set(int(x) for x in v) for k, v in raw.items()}


_SIG_CACHE: dict[int, set[int]] | None = None


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
        global _SIG_CACHE
        if _SIG_CACHE is None:
            _SIG_CACHE = _load_signal_set()

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
        sig_ts = _SIG_CACHE.get(day, set())
        in_sig = ts in sig_ts

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

        if j and in_sig:
            d53 = state.order_depths.get(S5300)
            if d53 and d53.buy_orders and d53.sell_orders:
                sp53 = tob_spread(d53)
                if sp53 is not None and sp53 >= _MIN_S5300_SPREAD:
                    last = int(bu.get("lst53", -10**9))
                    if ts - last >= _COOLDOWN_TS:
                        bbi = int(max(d53.buy_orders))
                        pos53 = int(state.position.get(S5300, 0) or 0)
                        lim53 = LIMITS[S5300]
                        if pos53 > -lim53:
                            q = min(_FADE_Q, lim53 + pos53)
                            if q > 0:
                                out[S5300].append(Order(S5300, bbi, -q))
                                bu["lst53"] = ts

        return out, 0, json.dumps({_TD: bu}, separators=(",", ":"))
