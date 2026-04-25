"""Neighbor vertical MR v3: v2 + local spline gap residual filter + 4-strike book cap.

SIGNAL (v2 unchanged for raw gaps)
  g_i = mid_i - mid_{i+1}, EWMA -> z1_i, entry |z1| >= Z_G (same 2.9 as v2).

SPLINE / LOCAL RESIDUAL (interior 1..7)
  R_i = g_i - 0.5 * (g_{i-1} + g_{i+1})
  (centered "bump" in the gap string).  Separate EWMA on R_{j+1} for j=0..6 -> z_R[j].
  For interior i, require: |z_R[i-1]| >= Z_R_MIN and z1[i] * z_R[i-1] > 0 (both signal the
  same local kink; filters edges where a big z1 is only a smooth slope across strikes).

  Edges i=0,8: no R term (v2 only).

BOOK
  _sum four half-spreads (strikes i-1..i+2) <= MAX_SPREAD_SUM.

GREEKS / T
  Same as v2 (day TTE, BS vega, full delta hedge on extract).
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
N_R = 7

LIMITS = {
    HYDRO: 200,
    UNDER: 200,
    **{s: 300 for s in SYMS},
}

ALPHA = 0.03
Z_G = 2.9
Z_R_MIN = 0.45
Z_SCALE = 2.8
BASE_SIZE = 8
MAX_PAIR = 14
MIN_TICKS = 85
TRADE_EVERY = 3
MAX_SPREAD_SUM = 100.0


def _tte_years(day_idx: int) -> float:
    d = max(0, min(day_idx, 2))
    tte_d = 8 - d
    return max(tte_d, 1) / 365.0


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


def _bs_delta_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * st)
    return _N.cdf(d1)


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


def _sum_spans(spreads: list[float], i: int) -> float:
    if i == 0:
        return spreads[0] + spreads[1] + spreads[2] + spreads[3]
    if i == 8:
        return spreads[5] + spreads[6] + spreads[7] + spreads[8]
    return spreads[i - 1] + spreads[i] + spreads[i + 1] + spreads[i + 2]


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            td = {}

        prev_ts = td.get("prev_ts")
        ts = state.timestamp
        if prev_ts is not None and ts < prev_ts:
            td["day_idx"] = int(td.get("day_idx", 0)) + 1
            td["m"] = [138.0] * N_EDGE
            td["v"] = [40000.0] * N_EDGE
            td["mR"] = [0.0] * N_R
            td["vR"] = [2500.0] * N_R
            td["ticks"] = 0
        td["prev_ts"] = ts

        ticks = int(td.get("ticks", 0)) + 1
        td["ticks"] = ticks

        day_idx = int(td.get("day_idx", 0))
        T = _tte_years(day_idx)

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
        mR_ew = td.get("mR", [0.0] * N_R)
        vR_ew = td.get("vR", [2500.0] * N_R)
        if len(m_ew) != N_EDGE:
            m_ew = [138.0] * N_EDGE
        if len(v_ew) != N_EDGE:
            v_ew = [40000.0] * N_EDGE
        if len(mR_ew) != N_R:
            mR_ew = [0.0] * N_R
        if len(vR_ew) != N_R:
            vR_ew = [2500.0] * N_R

        a = ALPHA
        z1 = [0.0] * N_EDGE
        for i in range(N_EDGE):
            g = gaps[i]
            mu_old = m_ew[i]
            var_old = v_ew[i]
            sig_old = max(math.sqrt(var_old), 1e-6)
            z1[i] = (g - mu_old) / sig_old
            resid = g - mu_old
            m_ew[i] = (1 - a) * mu_old + a * g
            v_ew[i] = max((1 - a) * var_old + a * (resid * resid), 1.0)

        zR = [0.0] * N_R
        for j in range(N_R):
            i = j + 1
            r = gaps[i] - 0.5 * (gaps[i - 1] + gaps[i + 1])
            mu0 = mR_ew[j]
            v0 = vR_ew[j]
            sig0 = max(math.sqrt(v0), 1e-6)
            zR[j] = (r - mu0) / sig0
            r2 = r - mu0
            mR_ew[j] = (1 - a) * mu0 + a * r
            vR_ew[j] = max((1 - a) * v0 + a * (r2 * r2), 1.0)

        td["m"] = m_ew
        td["v"] = v_ew
        td["mR"] = mR_ew
        td["vR"] = vR_ew

        if ticks < MIN_TICKS or (ticks % TRADE_EVERY) != 0:
            return orders, 0, json.dumps(td)

        best_i = -1
        best_absz = 0.0
        best_z = 0.0
        for i in range(N_EDGE):
            if abs(z1[i]) < Z_G:
                continue
            if 1 <= i <= 7:
                zr = zR[i - 1]
                if abs(zr) < Z_R_MIN or (z1[i] * zr) <= 0.0:
                    continue
            if _sum_spans(spreads, i) > MAX_SPREAD_SUM:
                continue
            az = abs(z1[i])
            if az > best_absz:
                best_absz = az
                best_z = z1[i]
                best_i = i

        if best_i < 0:
            return orders, 0, json.dumps(td)

        hi_sym = SYMS[best_i]
        lo_sym = SYMS[best_i + 1]
        Kh, Kl = float(STRIKES[best_i]), float(STRIKES[best_i + 1])

        iv_h = _implied_vol(mids[best_i], Su, Kh, T)
        iv_l = _implied_vol(mids[best_i + 1], Su, Kl, T)
        sig_ref = 0.25
        sh = iv_h if iv_h else sig_ref
        sl = iv_l if iv_l else sig_ref
        vh = _bs_vega(Su, Kh, T, sh)
        vl = _bs_vega(Su, Kl, T, sl)
        vega_norm = max((vh + vl) / 2.0, 1.0)

        dh = _bs_delta_call(Su, Kh, T, sh)
        dl = _bs_delta_call(Su, Kl, T, sl)

        two_leg = spreads[best_i] + spreads[best_i + 1]
        edge_scale = max(two_leg, 2.0)
        zfac = min(best_absz / Z_SCALE, 2.0)
        raw = int(BASE_SIZE * zfac * (200.0 / vega_norm) * (10.0 / edge_scale))
        qty = max(1, min(MAX_PAIR, raw))

        pos_hi = state.position.get(hi_sym, 0)
        pos_lo = state.position.get(lo_sym, 0)
        pos_u = state.position.get(UNDER, 0)

        if best_z > 0:
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
            net_d = q * (dl - dh)
        else:
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
            net_d = q * (dh - dl)

        if UNDER in depths:
            du = depths[UNDER]
            hedge_q = int(round(net_d))
            if hedge_q != 0:
                if hedge_q > 0:
                    hq = min(hedge_q, LIMITS[UNDER] + pos_u)
                    if hq > 0 and du.buy_orders:
                        bid_u = max(du.buy_orders)
                        orders[UNDER].append(Order(UNDER, bid_u, -hq))
                else:
                    need = -hedge_q
                    hq = min(need, LIMITS[UNDER] - pos_u)
                    if hq > 0 and du.sell_orders:
                        ask_u = min(du.sell_orders)
                        orders[UNDER].append(Order(UNDER, ask_u, hq))

        return orders, 0, json.dumps(td)
