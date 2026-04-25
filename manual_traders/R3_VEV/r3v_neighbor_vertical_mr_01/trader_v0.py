"""Neighbor vertical mean-reversion: EWMA z-score on adjacent-strike mid gaps.

TTE assumption (same as analysis.json / round3description example mapping to CSV days 0–2):
use fixed T = 7/365 years for in-trader BS vega scaling only (historical tapes are 8/7/6d;
vega is used as a relative weight, not for absolute pricing).
"""
from __future__ import annotations

import json
import math
from datamodel import Order, TradingState
from statistics import NormalDist

_N = NormalDist()

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMS = [f"VEV_{k}" for k in STRIKES]
UNDER = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
N_EDGE = len(STRIKES) - 1

LIMITS = {
    HYDRO: 200,
    UNDER: 200,
    **{s: 300 for s in SYMS},
}

# --- tunables (iteration 0) ---
ALPHA = 0.02
Z_THRESHOLD = 1.8
Z_SCALE = 2.5
BASE_SIZE = 12
MAX_PAIR = 22
MIN_TICKS = 55
VEGA_T_YEARS = 7.0 / 365.0


def _mid(depth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders)
    ba = min(depth.sell_orders)
    return 0.5 * (bb + ba)


def _spread_half(depth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders)
    ba = min(depth.sell_orders)
    return 0.5 * (ba - bb)


def _bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * st)
    return S * st * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)


def _implied_vol(mid: float, S: float, K: float, T: float) -> float | None:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-9 or T <= 0:
        return None
    lo, hi = 1e-4, 4.0
    for _ in range(45):
        m = 0.5 * (lo + hi)
        # Black–Scholes call with r=0
        st = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * m * m * T) / (m * st)
        d2 = d1 - m * st
        pr = S * _N.cdf(d1) - K * _N.cdf(d2)
        if pr > mid:
            hi = m
        else:
            lo = m
    sig = 0.5 * (lo + hi)
    return sig if sig > 1e-4 else None


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}
        ticks = int(td.get("ticks", 0)) + 1
        td["ticks"] = ticks

        orders: dict[str, list[Order]] = {}
        for p in LIMITS:
            orders[p] = []

        depths = state.order_depths
        Su = _mid(depths.get(UNDER)) if UNDER in depths else None
        if Su is None or Su <= 0:
            return orders, 0, json.dumps(td)

        mids: list[float] = []
        spreads: list[float] = []
        for s in SYMS:
            d = depths.get(s)
            if d is None:
                return orders, 0, json.dumps(td)
            m = _mid(d)
            h = _spread_half(d)
            if m is None or h is None:
                return orders, 0, json.dumps(td)
            mids.append(m)
            spreads.append(h)

        gaps = [mids[i] - mids[i + 1] for i in range(N_EDGE)]

        m_ew = td.get("m", [138.0] * N_EDGE)
        v_ew = td.get("v", [40000.0] * N_EDGE)
        if len(m_ew) != N_EDGE:
            m_ew = [138.0] * N_EDGE
        if len(v_ew) != N_EDGE:
            v_ew = [40000.0] * N_EDGE

        a = ALPHA
        z_scores = [0.0] * N_EDGE
        for i in range(N_EDGE):
            g = gaps[i]
            mu_old = m_ew[i]
            var_old = v_ew[i]
            sig_old = max(math.sqrt(var_old), 1e-6)
            z_scores[i] = (g - mu_old) / sig_old
            resid = g - mu_old
            m_ew[i] = (1 - a) * mu_old + a * g
            v_ew[i] = max((1 - a) * var_old + a * (resid * resid), 1.0)
        td["m"] = m_ew
        td["v"] = v_ew

        if ticks < MIN_TICKS:
            return orders, 0, json.dumps(td)

        best_i = -1
        best_absz = 0.0
        best_z = 0.0
        for i in range(N_EDGE):
            z = z_scores[i]
            az = abs(z)
            if az > best_absz:
                best_absz = az
                best_z = z
                best_i = i

        if best_i < 0 or best_absz < Z_THRESHOLD:
            return orders, 0, json.dumps(td)

        hi_sym = SYMS[best_i]
        lo_sym = SYMS[best_i + 1]
        Kh, Kl = float(STRIKES[best_i]), float(STRIKES[best_i + 1])

        iv_h = _implied_vol(mids[best_i], Su, Kh, VEGA_T_YEARS)
        iv_l = _implied_vol(mids[best_i + 1], Su, Kl, VEGA_T_YEARS)
        sig_ref = 0.25
        vh = _bs_vega(Su, Kh, VEGA_T_YEARS, iv_h if iv_h else sig_ref)
        vl = _bs_vega(Su, Kl, VEGA_T_YEARS, iv_l if iv_l else sig_ref)
        vega_norm = max((vh + vl) / 2.0, 1.0)

        two_leg = spreads[best_i] + spreads[best_i + 1]
        edge_scale = max(two_leg, 2.0)
        zfac = min(best_absz / Z_SCALE, 2.0)
        raw = int(BASE_SIZE * zfac * (200.0 / vega_norm) * (10.0 / edge_scale))
        qty = max(1, min(MAX_PAIR, raw))

        pos_hi = state.position.get(hi_sym, 0)
        pos_lo = state.position.get(lo_sym, 0)

        if best_z > 0:
            # gap high: sell hi strike, buy lo strike
            sell_hi = min(qty, LIMITS[hi_sym] + pos_hi)
            buy_lo = min(qty, LIMITS[lo_sym] - pos_lo)
            q = min(sell_hi, buy_lo)
            if q <= 0:
                return orders, 0, json.dumps(td)
            d_hi = depths[hi_sym]
            d_lo = depths[lo_sym]
            if not d_hi.buy_orders or not d_lo.sell_orders:
                return orders, 0, json.dumps(td)
            bid_hi = max(d_hi.buy_orders)
            ask_lo = min(d_lo.sell_orders)
            orders[hi_sym].append(Order(hi_sym, bid_hi, -q))
            orders[lo_sym].append(Order(lo_sym, ask_lo, q))
        else:
            # gap low: buy hi strike, sell lo strike
            buy_hi = min(qty, LIMITS[hi_sym] - pos_hi)
            sell_lo = min(qty, LIMITS[lo_sym] + pos_lo)
            q = min(buy_hi, sell_lo)
            if q <= 0:
                return orders, 0, json.dumps(td)
            d_hi = depths[hi_sym]
            d_lo = depths[lo_sym]
            if not d_hi.sell_orders or not d_lo.buy_orders:
                return orders, 0, json.dumps(td)
            ask_hi = min(d_hi.sell_orders)
            bid_lo = max(d_lo.buy_orders)
            orders[hi_sym].append(Order(hi_sym, ask_hi, q))
            orders[lo_sym].append(Order(lo_sym, bid_lo, -q))

        return orders, 0, json.dumps(td)
