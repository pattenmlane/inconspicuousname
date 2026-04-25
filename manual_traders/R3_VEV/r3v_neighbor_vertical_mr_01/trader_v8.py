"""Family10 pivot: smile_quadratic_logm_wls with extrinsic cutoff.

Model:
- Fit implied vol smile: iv(x)=a+b*x+c*x^2 where x=log(K/S)
- Weighted LS with w=1/spread^2 on points whose extrinsic >= EXTRINSIC_CUTOFF
- Predict fair option prices by BS from fitted iv for all strikes
- Trade residuals (mid - fair) with z-score on EWMA residual per symbol

Variation requirement:
- Keep all strikes in tradable universe, but filter tiny-extrinsic points from fit only.
"""
from __future__ import annotations

import json
import math
from datamodel import Order, TradingState
from statistics import NormalDist

N = NormalDist()

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
SYMS = [f"VEV_{k}" for k in STRIKES]
UNDER = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"

LIMITS = {
    UNDER: 200,
    HYDRO: 200,
    **{s: 300 for s in SYMS},
}

# tuned around analysis_family10_extrinsic_cutoff.json
EXTRINSIC_CUTOFF = 0.5
WARMUP = 90
TRADE_EVERY = 5
ALPHA = 0.03
OPEN_Z = 3.2
CLOSE_Z = 0.7
MAX_CLIP = 8
MISPRICING_SCALE = 10.0


def _tte_years(day_idx: int) -> float:
    return max(8 - max(0, min(day_idx, 2)), 1) / 365.0


def _mid(depth):
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return 0.5 * (max(depth.buy_orders) + min(depth.sell_orders))


def _half_spread(depth):
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return 0.5 * (min(depth.sell_orders) - max(depth.buy_orders))


def _bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0 or S <= 0:
        return max(S - K, 0.0)
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * st)
    d2 = d1 - sig * st
    return S * N.cdf(d1) - K * N.cdf(d2)


def _bs_delta(S, K, T, sig):
    if T <= 0 or sig <= 0 or S <= 0:
        return 0.0
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * st)
    return N.cdf(d1)


def _iv(mid, S, K, T):
    ins = max(S - K, 0.0)
    if mid <= ins + 1e-9 or T <= 0:
        return None
    lo, hi = 1e-4, 4.0
    for _ in range(45):
        m = 0.5 * (lo + hi)
        p = _bs_call(S, K, T, m)
        if p > mid:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def _fit_quad_wls(xs, ys, ws):
    # solve normal equations for [1, x, x^2]
    s00 = s01 = s02 = s11 = s12 = s22 = 0.0
    t0 = t1 = t2 = 0.0
    for x, y, w in zip(xs, ys, ws):
        x2 = x * x
        s00 += w
        s01 += w * x
        s02 += w * x2
        s11 += w * x * x
        s12 += w * x2 * x
        s22 += w * x2 * x2
        t0 += w * y
        t1 += w * x * y
        t2 += w * x2 * y
    A = [
        [s00, s01, s02, t0],
        [s01, s11, s12, t1],
        [s02, s12, s22, t2],
    ]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(A[r][i]))
        if abs(A[piv][i]) < 1e-12:
            return None
        A[i], A[piv] = A[piv], A[i]
        p = A[i][i]
        for c in range(i, 4):
            A[i][c] /= p
        for r in range(3):
            if r == i:
                continue
            f = A[r][i]
            for c in range(i, 4):
                A[r][c] -= f * A[i][c]
    return (A[0][3], A[1][3], A[2][3])


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
            td["ticks"] = 0
            td["res_mu"] = {s: 0.0 for s in SYMS}
            td["res_var"] = {s: 100.0 for s in SYMS}
        td["prev_ts"] = ts
        ticks = int(td.get("ticks", 0)) + 1
        td["ticks"] = ticks
        day_idx = int(td.get("day_idx", 0))

        orders = {p: [] for p in LIMITS}
        depths = state.order_depths

        if UNDER not in depths:
            return orders, 0, json.dumps(td)
        S = _mid(depths[UNDER])
        if S is None or S <= 0:
            return orders, 0, json.dumps(td)

        T = _tte_years(day_idx)
        mids = {}
        hs = {}
        ivs = {}
        xs = {}

        for sym, K in zip(SYMS, STRIKES):
            d = depths.get(sym)
            if d is None:
                return orders, 0, json.dumps(td)
            m = _mid(d)
            h = _half_spread(d)
            if m is None or h is None:
                return orders, 0, json.dumps(td)
            mids[sym] = m
            hs[sym] = max(h, 0.5)
            x = math.log(K / S)
            xs[sym] = x
            ivs[sym] = _iv(m, S, float(K), T)

        # fit subset with extrinsic cutoff
        fx, fy, fw = [], [], []
        for sym, K in zip(SYMS, STRIKES):
            iv = ivs[sym]
            if iv is None:
                continue
            extr = mids[sym] - max(S - K, 0.0)
            if extr < EXTRINSIC_CUTOFF:
                continue
            fx.append(xs[sym])
            fy.append(iv)
            fw.append(1.0 / (hs[sym] * hs[sym]))

        if len(fx) < 4:
            return orders, 0, json.dumps(td)

        coefs = _fit_quad_wls(fx, fy, fw)
        if coefs is None:
            return orders, 0, json.dumps(td)
        a, b, c = coefs

        res_mu = td.get("res_mu", {s: 0.0 for s in SYMS})
        res_var = td.get("res_var", {s: 100.0 for s in SYMS})
        # ensure keys
        for s in SYMS:
            res_mu.setdefault(s, 0.0)
            res_var.setdefault(s, 100.0)

        # update residual states for all symbols
        residuals = {}
        for sym, K in zip(SYMS, STRIKES):
            x = xs[sym]
            iv_fit = max(0.01, a + b * x + c * x * x)
            fair = _bs_call(S, float(K), T, iv_fit)
            residuals[sym] = mids[sym] - fair
            r = residuals[sym]
            mu = float(res_mu[sym])
            var = float(res_var[sym])
            res_mu[sym] = (1 - ALPHA) * mu + ALPHA * r
            dv = r - mu
            res_var[sym] = max((1 - ALPHA) * var + ALPHA * dv * dv, 1.0)

        td["res_mu"] = res_mu
        td["res_var"] = res_var

        if ticks < WARMUP or (ticks % TRADE_EVERY) != 0:
            return orders, 0, json.dumps(td)

        # pick strongest mispricing zscore across symbols
        best_sym = None
        best_z = 0.0
        for sym in SYMS:
            mu = float(res_mu[sym])
            sd = math.sqrt(float(res_var[sym]))
            z = (residuals[sym] - mu) / max(sd, 1e-6)
            if abs(z) > abs(best_z):
                best_z = z
                best_sym = sym

        if best_sym is None:
            return orders, 0, json.dumps(td)

        # optional close condition: if current symbol position and z close to mean, flatten some
        pos = state.position.get(best_sym, 0)
        if abs(best_z) < CLOSE_Z and pos != 0:
            d = depths[best_sym]
            if pos > 0 and d.buy_orders:
                q = min(abs(pos), MAX_CLIP)
                orders[best_sym].append(Order(best_sym, max(d.buy_orders), -q))
            elif pos < 0 and d.sell_orders:
                q = min(abs(pos), MAX_CLIP)
                orders[best_sym].append(Order(best_sym, min(d.sell_orders), q))
            return orders, 0, json.dumps(td)

        if abs(best_z) < OPEN_Z:
            return orders, 0, json.dumps(td)

        d = depths[best_sym]
        K = STRIKES[SYMS.index(best_sym)]
        x = xs[best_sym]
        iv_fit = max(0.01, a + b * x + c * x * x)
        delta = _bs_delta(S, float(K), T, iv_fit)

        # size from z and local spread
        zfac = min(abs(best_z) / MISPRICING_SCALE, 2.0)
        qty = max(1, min(MAX_CLIP, int(MAX_CLIP * zfac * (2.0 / hs[best_sym]))))

        pos_opt = state.position.get(best_sym, 0)
        pos_u = state.position.get(UNDER, 0)

        if best_z > 0:
            # overvalued -> sell option
            q = min(qty, LIMITS[best_sym] + pos_opt)
            if q > 0 and d.buy_orders:
                orders[best_sym].append(Order(best_sym, max(d.buy_orders), -q))
                hedge = int(round(-q * delta))
                if hedge > 0 and depths[UNDER].sell_orders:
                    bq = min(hedge, LIMITS[UNDER] - pos_u)
                    if bq > 0:
                        orders[UNDER].append(Order(UNDER, min(depths[UNDER].sell_orders), bq))
                elif hedge < 0 and depths[UNDER].buy_orders:
                    sq = min(-hedge, LIMITS[UNDER] + pos_u)
                    if sq > 0:
                        orders[UNDER].append(Order(UNDER, max(depths[UNDER].buy_orders), -sq))
        else:
            # undervalued -> buy option
            q = min(qty, LIMITS[best_sym] - pos_opt)
            if q > 0 and d.sell_orders:
                orders[best_sym].append(Order(best_sym, min(d.sell_orders), q))
                hedge = int(round(-q * delta))
                if hedge > 0 and depths[UNDER].sell_orders:
                    bq = min(hedge, LIMITS[UNDER] - pos_u)
                    if bq > 0:
                        orders[UNDER].append(Order(UNDER, min(depths[UNDER].sell_orders), bq))
                elif hedge < 0 and depths[UNDER].buy_orders:
                    sq = min(-hedge, LIMITS[UNDER] + pos_u)
                    if sq > 0:
                        orders[UNDER].append(Order(UNDER, max(depths[UNDER].buy_orders), -sq))

        return orders, 0, json.dumps(td)
