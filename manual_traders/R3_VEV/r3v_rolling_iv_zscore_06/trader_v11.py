"""
v11: family-13 realized_vol_regime (RV vs ATM IV spread) with hysteresis.

Action = quote-width scaling only on VEV_5000:
- Neutral regime: narrower quote width
- High-RV regime: wider quote width

No directional market-taking logic; both bid and ask are always quoted.
"""

from __future__ import annotations

import json
import math
from collections import deque
from datamodel import Order, OrderDepth, TradingState

VEV_TARGET = "VEV_5000"
UNDER = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"

LIMITS = {
    HYDRO: 200,
    UNDER: 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

STRIKE = 5000
T_YEAR = 7.0 / 365.0

RV_WIN = 40
DT_YEAR = 1.0 / (365.0 * 10000.0)

ENTER_HIGH = 0.24
EXIT_HIGH = 0.19
ENTER_LOW = -0.24
EXIT_LOW = -0.19

NEUTRAL_HALF_WIDTH = 1
HIGH_HALF_WIDTH = 3
LOW_HALF_WIDTH = 1
MAX_MAKE_Q = 10


def _N_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(spot: float, strike: float, t: float, vol: float) -> float:
    if t <= 0 or vol <= 0:
        return max(spot - strike, 0.0)
    sig_rt = vol * math.sqrt(t)
    if sig_rt < 1e-12:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / sig_rt
    d2 = d1 - sig_rt
    return spot * _N_cdf(d1) - strike * _N_cdf(d2)


def implied_vol(spot: float, strike: float, t: float, price: float) -> float | None:
    intrinsic = max(spot - strike, 0.0)
    if price <= intrinsic + 1e-6:
        return None
    lo, hi = 1e-4, 4.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        th = bs_call_price(spot, strike, t, mid)
        if th > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _touch(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}

        ret_hist: deque[float] = deque((float(x) for x in td.get("_ret", [])), maxlen=RV_WIN + 10)
        regime = int(td.get("_rg", 0))
        prev_s = td.get("_ps", None)

        result: dict[str, list[Order]] = {p: [] for p in LIMITS}

        und = state.order_depths.get(UNDER)
        vev = state.order_depths.get(VEV_TARGET)
        if und is None or vev is None:
            td["_ret"] = list(ret_hist)
            td["_rg"] = regime
            td["_ps"] = prev_s
            return result, 0, json.dumps(td)

        ub, ua = _touch(und)
        vb, va = _touch(vev)
        if ub is None or ua is None or vb is None or va is None:
            td["_ret"] = list(ret_hist)
            td["_rg"] = regime
            td["_ps"] = prev_s
            return result, 0, json.dumps(td)

        s_mid = 0.5 * (ub + ua)
        c_mid = 0.5 * (vb + va)

        if isinstance(prev_s, (int, float)) and prev_s > 0 and s_mid > 0:
            ret_hist.append(math.log(s_mid / float(prev_s)))
        prev_s = s_mid

        spread = None
        rv_now = None
        iv_now = implied_vol(s_mid, STRIKE, T_YEAR, c_mid)
        if len(ret_hist) >= RV_WIN and iv_now is not None:
            rv_now = math.sqrt(sum(r * r for r in ret_hist) / len(ret_hist)) / math.sqrt(DT_YEAR)
            spread = rv_now - iv_now

            if regime == 0:
                if spread >= ENTER_HIGH:
                    regime = 1
                elif spread <= ENTER_LOW:
                    regime = -1
            elif regime == 1:
                if spread <= EXIT_HIGH:
                    regime = 0
            else:
                if spread >= EXIT_LOW:
                    regime = 0

        if regime == 1:
            half_width = HIGH_HALF_WIDTH
        elif regime == -1:
            half_width = LOW_HALF_WIDTH
        else:
            half_width = NEUTRAL_HALF_WIDTH

        # Width-scaling only around midpoint of current touch.
        mid = 0.5 * (vb + va)
        bid_px = int(math.floor(mid - half_width))
        ask_px = int(math.ceil(mid + half_width))

        # Ensure passive non-crossing quotes.
        bid_px = min(bid_px, va - 1)
        ask_px = max(ask_px, vb + 1)

        pos = state.position.get(VEV_TARGET, 0)
        lim = LIMITS[VEV_TARGET]
        buy_q = min(MAX_MAKE_Q, max(0, lim - pos))
        sell_q = min(MAX_MAKE_Q, max(0, lim + pos))

        if buy_q > 0:
            result[VEV_TARGET].append(Order(VEV_TARGET, bid_px, buy_q))
        if sell_q > 0:
            result[VEV_TARGET].append(Order(VEV_TARGET, ask_px, -sell_q))

        td["_ret"] = list(ret_hist)[-RV_WIN - 5 :]
        td["_rg"] = regime
        td["_ps"] = prev_s
        td["_rv"] = rv_now if rv_now is not None else 0.0
        td["_iv"] = iv_now if iv_now is not None else 0.0
        td["_sp"] = spread if spread is not None else 0.0
        return result, 0, json.dumps(td)
