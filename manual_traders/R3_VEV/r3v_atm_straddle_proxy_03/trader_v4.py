"""
Iteration 4: ATM 5000/5100 bundle with conservative IV-shock trigger.

Same thesis: coordinated ATM bundle legs + extract hedge on IV/extract shocks.
Adds spread gate + stronger shock threshold to reduce overtrading seen in v2/v3.
TTE mapping from round3description: historical day d -> TTE = 8-d days.
"""
from __future__ import annotations

import json
import math
from datamodel import Order, OrderDepth, TradingState
from statistics import NormalDist

_N = NormalDist()

HYDROGEL = "HYDROGEL_PACK"
EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_5000 = "VEV_5000"
VEV_5100 = "VEV_5100"
K0, K1 = 5000.0, 5100.0

POS_LIMITS = {
    HYDROGEL: 200,
    EXTRACT: 200,
    VEV_5000: 300,
    VEV_5100: 300,
}

MISPRICE_EDGE = 3.0
RV_WINDOW = 50
HYDRO_EMA_HALFLIFE = 200
HYDRO_FADE_EDGE = 12
ORDER_CLIP = 10
DAYS_PER_YEAR = 365.0
STEPS_PER_SIM_DAY = 10000
SHOCK_ABS_EDGE = 0.06
SPREAD_GATE_SUM = 10.0


def _mid(depth: OrderDepth) -> tuple[float, int, int] | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders)
    ba = min(depth.sell_orders)
    return (bb + ba) / 2.0, bb, ba


def _spread(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders)
    ba = min(depth.sell_orders)
    return float(ba - bb)


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 1e-8 or S <= 0:
        return max(S - K, 0.0)
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * st)
    d2 = d1 - sigma * st
    return S * _N.cdf(d1) - K * _N.cdf(d2)


def _bs_delta_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 1e-8 or S <= 0:
        return 1.0 if S > K else 0.0
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * st)
    return _N.cdf(d1)


def _implied_vol(mid: float, S: float, K: float, T: float) -> float:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-9:
        return 0.08
    lo, hi = 0.02, 2.5
    for _ in range(48):
        sig = 0.5 * (lo + hi)
        p = _bs_call(S, K, T, sig)
        if p > mid:
            hi = sig
        else:
            lo = sig
    return max(0.06, min(1.2, 0.5 * (lo + hi)))


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        day_idx = int(getattr(state, "_prosperity4bt_hist_day", 0))
        tte_days = 8.0 - float(min(max(day_idx, 0), 2))
        T = max(tte_days / DAYS_PER_YEAR, 1e-6)

        ex = state.order_depths.get(EXTRACT)
        d0 = state.order_depths.get(VEV_5000)
        d1 = state.order_depths.get(VEV_5100)
        dh = state.order_depths.get(HYDROGEL)

        out: dict[str, list[Order]] = {k: [] for k in POS_LIMITS}

        if ex is None or d0 is None or d1 is None:
            return out, 0, json.dumps(td)

        m_ex = _mid(ex)
        m0 = _mid(d0)
        m1 = _mid(d1)
        if m_ex is None or m0 is None or m1 is None:
            return out, 0, json.dumps(td)

        S, _, _ = m_ex
        mid0, bb0, ba0 = m0
        mid1, bb1, ba1 = m1

        hist = td.get("_ex_hist", [])
        hist.append(float(S))
        if len(hist) > RV_WINDOW + 5:
            hist = hist[-(RV_WINDOW + 5) :]
        td["_ex_hist"] = hist

        sigma_rv = 0.22
        if len(hist) >= 5:
            rets = []
            for i in range(1, len(hist)):
                a, b = hist[i - 1], hist[i]
                if a > 0 and b > 0:
                    rets.append(math.log(b / a))
            if len(rets) >= 4:
                mu = sum(rets) / len(rets)
                var = sum((x - mu) ** 2 for x in rets) / (len(rets) - 1)
                vol_tick = math.sqrt(max(var, 1e-16))
                sigma_rv = max(0.08, min(0.9, vol_tick * math.sqrt(STEPS_PER_SIM_DAY * DAYS_PER_YEAR)))

        iv0 = _implied_vol(mid0, S, K0, T)
        iv1 = _implied_vol(mid1, S, K1, T)
        theo0 = _bs_call(S, K0, T, iv0)
        theo1 = _bs_call(S, K1, T, iv1)
        theo_bundle = theo0 + theo1
        bundle_mid = mid0 + mid1
        mis = bundle_mid - theo_bundle

        del0 = _bs_delta_call(S, K0, T, iv0)
        del1 = _bs_delta_call(S, K1, T, iv1)
        unit_straddle_delta = del0 + del1

        # IV shock vs realized: fade rich straddle when ATM IVs dominate RV (coordinated short).
        iv_avg = 0.5 * (iv0 + iv1)
        shock = iv_avg - sigma_rv

        p0 = state.position.get(VEV_5000, 0)
        p1 = state.position.get(VEV_5100, 0)
        px = state.position.get(EXTRACT, 0)
        ph = state.position.get(HYDROGEL, 0)

        sp0 = _spread(d0) or 99.0
        sp1 = _spread(d1) or 99.0
        spread_ok = (sp0 + sp1) <= SPREAD_GATE_SUM

        want_short = mis > MISPRICE_EDGE or shock > SHOCK_ABS_EDGE
        want_long = mis < -MISPRICE_EDGE or shock < -SHOCK_ABS_EDGE
        trade_short_bundle = want_short and (not want_long or mis > 0)
        trade_long_bundle = want_long and (not want_short or mis < 0)

        if spread_ok and trade_short_bundle:
            q = min(ORDER_CLIP, POS_LIMITS[VEV_5000] + p0, POS_LIMITS[VEV_5100] + p1)
            q = max(0, q)
            if q > 0:
                out[VEV_5000].append(Order(VEV_5000, bb0, -q))
                out[VEV_5100].append(Order(VEV_5100, bb1, -q))
            hedge = int(round(unit_straddle_delta * q))
            if hedge > 0 and px < POS_LIMITS[EXTRACT]:
                hq = min(hedge, POS_LIMITS[EXTRACT] - px, ORDER_CLIP * 3)
                if hq > 0:
                    out[EXTRACT].append(Order(EXTRACT, int(math.ceil(S)), hq))
            elif hedge < 0 and px > -POS_LIMITS[EXTRACT]:
                hq = min(-hedge, POS_LIMITS[EXTRACT] + px, ORDER_CLIP * 3)
                if hq > 0:
                    out[EXTRACT].append(Order(EXTRACT, int(math.floor(S)), -hq))

        elif spread_ok and trade_long_bundle:
            q = min(ORDER_CLIP, POS_LIMITS[VEV_5000] - p0, POS_LIMITS[VEV_5100] - p1)
            q = max(0, q)
            if q > 0:
                out[VEV_5000].append(Order(VEV_5000, ba0, q))
                out[VEV_5100].append(Order(VEV_5100, ba1, q))
            hedge = int(round(unit_straddle_delta * q))
            if hedge > 0 and px > -POS_LIMITS[EXTRACT]:
                hq = min(hedge, POS_LIMITS[EXTRACT] + px, ORDER_CLIP * 3)
                if hq > 0:
                    out[EXTRACT].append(Order(EXTRACT, int(math.floor(S)), -hq))
            elif hedge < 0 and px < POS_LIMITS[EXTRACT]:
                hq = min(-hedge, POS_LIMITS[EXTRACT] - px, ORDER_CLIP * 3)
                if hq > 0:
                    out[EXTRACT].append(Order(EXTRACT, int(math.ceil(S)), hq))

        if dh is not None:
            m_h = _mid(dh)
            sp_h = _spread(dh)
            if m_h and sp_h is not None and sp_h <= 20:
                ema = float(td.get("_hyd_ema", m_h[0]))
                a = 2.0 / (HYDRO_EMA_HALFLIFE + 1.0)
                ema = (1 - a) * ema + a * m_h[0]
                td["_hyd_ema"] = ema
                dev = m_h[0] - ema
                if abs(dev) > HYDRO_FADE_EDGE:
                    q = min(8, POS_LIMITS[HYDROGEL] - ph) if dev < 0 else min(8, POS_LIMITS[HYDROGEL] + ph)
                    q = max(0, q)
                    if q > 0:
                        if dev < 0:
                            out[HYDROGEL].append(Order(HYDROGEL, int(math.ceil(m_h[0])), q))
                        else:
                            out[HYDROGEL].append(Order(HYDROGEL, int(math.floor(m_h[0])), -q))

        return out, 0, json.dumps(td)
