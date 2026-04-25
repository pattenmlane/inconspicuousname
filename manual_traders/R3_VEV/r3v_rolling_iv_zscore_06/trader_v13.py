"""
v13: family-13 realized_vol_regime (distinct from v11/v12) with hysteresis +
quote-width scaling only, plus regime-aware quote-center skew.

Signal:
  ratio = RV_short / ATM_IV - 1
  RV_short from VELVETFRUIT_EXTRACT log returns (window=30), annualized.
  ATM_IV from BS implied vol of VEV_5000 using mids.

Regime hysteresis:
  neutral -> high when ratio >= ENTER_HIGH
  high -> neutral when ratio <= EXIT_HIGH
  (low regime included for completeness, rarely visited on this tape)

Action (still width-scaling-only maker):
  - Set half-width by regime
  - Shift quote center by +/- CENTER_SKEW ticks using the sign of the latest
    underlying move (if high RV and spot just dropped, bias center lower; if
    high RV and spot just rose, bias center higher).
  - Place passive bid/ask only; no directional market taking.
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

RV_WIN = 30
DT_YEAR = 1.0 / (365.0 * 10000.0)

ENTER_HIGH = 0.45
EXIT_HIGH = 0.28
ENTER_LOW = -0.30
EXIT_LOW = -0.15

NEUTRAL_HALF_WIDTH = 1
HIGH_HALF_WIDTH = 3
LOW_HALF_WIDTH = 1
CENTER_SKEW = 1
MAX_MAKE_Q = 8


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

        dspot = 0.0
        if isinstance(prev_s, (int, float)) and prev_s > 0 and s_mid > 0:
            ret_hist.append(math.log(s_mid / float(prev_s)))
            dspot = s_mid - float(prev_s)
        prev_s = s_mid

        ratio = None
        rv_now = None
        iv_now = implied_vol(s_mid, STRIKE, T_YEAR, c_mid)
        if len(ret_hist) >= RV_WIN and iv_now is not None and iv_now > 1e-9:
            rv_now = math.sqrt(sum(r * r for r in ret_hist) / len(ret_hist)) / math.sqrt(DT_YEAR)
            ratio = rv_now / iv_now - 1.0

            if regime == 0:
                if ratio >= ENTER_HIGH:
                    regime = 1
                elif ratio <= ENTER_LOW:
                    regime = -1
            elif regime == 1:
                if ratio <= EXIT_HIGH:
                    regime = 0
            else:
                if ratio >= EXIT_LOW:
                    regime = 0

        if regime == 1:
            half_width = HIGH_HALF_WIDTH
        elif regime == -1:
            half_width = LOW_HALF_WIDTH
        else:
            half_width = NEUTRAL_HALF_WIDTH

        # Width-scaling only + center skew (still pure quoting action)
        center = 0.5 * (vb + va)
        if regime == 1:
            if dspot < 0:
                center -= CENTER_SKEW
            elif dspot > 0:
                center += CENTER_SKEW

        bid_px = int(math.floor(center - half_width))
        ask_px = int(math.ceil(center + half_width))
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
        td["_ratio"] = ratio if ratio is not None else 0.0
        return result, 0, json.dumps(td)
