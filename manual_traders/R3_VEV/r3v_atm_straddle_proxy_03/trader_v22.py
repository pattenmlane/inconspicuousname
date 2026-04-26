"""
Iteration 22: v21 + shared joint tight-book signal (round3work/vouchers_final_strategy) and extract/VEV focus.

Per STRATEGY.txt: when VEV_5200 and VEV_5300 both have L1 spread <= 2 at the same tick, forward extract mid
dynamics are more favorable (see outputs/r3_tight_spread_summary.txt). Use that as a risk-on *sizing* signal:
raise bundle clip 4->5 and extract hedge cap when the joint gate is on; do not require it to trade (hard gate
was too sparse). Hydrogel disabled. TTE: hist day d -> (8-d) days.
"""
from __future__ import annotations

import json
import math
import numpy as np
from datamodel import Order, OrderDepth, TradingState
from statistics import NormalDist

_N = NormalDist()

HYDROGEL = "HYDROGEL_PACK"
EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_5000 = "VEV_5000"
VEV_5100 = "VEV_5100"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
K0, K1 = 5000.0, 5100.0

POS_LIMITS = {
    HYDROGEL: 200,
    EXTRACT: 200,
    VEV_5000: 300,
    VEV_5100: 300,
}

MISPRICE_EDGE = 25.0  # unused; vega+spread floor below
RV_WINDOW = 50
ORDER_CLIP = 4
ORDER_CLIP_JOINT_TIGHT = 5
TIGHT_5200_5300_TH = 2.0
DAYS_PER_YEAR = 365.0
STEPS_PER_SIM_DAY = 10000
SHOCK_ABS_EDGE = 0.20
SPREAD_GATE_SUM = 9.0
VEGA_MISPRICE_BASE = 15.0
VEGA_MISPRICE_PER_VEGA = 0.04
SPREAD_SUM_REF = 10.0
SPREAD_FLOOR_COEF = 0.4
SHOCK_SPREAD_COEF = 0.02
SHOCK_DS_REF = 2.5
SHOCK_DS_SCALE = 0.32
# BS straddle sum(gamma) is O(1e-3) here; scale so the term is comparable to vega+spread pieces (~10–30).
GAMMA_MISPRICE_PER_GAMMA = 5000.0
SHOCK_ONLY_MIN_DS = 0.5
SHOCK_WIDE_SS = 11.0
SHOCK_WIDE_MULT = 1.25


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


def _bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 1e-8 or S <= 0:
        return 0.0
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * st)
    return S * st * _N.pdf(d1)


def _bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 1e-8 or S <= 0:
        return 0.0
    st = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * st)
    return _N.pdf(d1) / (S * st * sigma)


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




def _fit_smile_quadratic_logm(
    S: float,
    T: float,
    mids: dict[str, float],
    spreads: dict[str, float],
    robust: bool,
):
    strikes = [4000.0, 4500.0, 5000.0, 5100.0, 5200.0, 5300.0, 5400.0, 5500.0, 6000.0, 6500.0]
    syms = [f"VEV_{int(k)}" for k in strikes]
    xs, ys, ws = [], [], []
    for sym, K in zip(syms, strikes):
        m = mids.get(sym)
        if m is None:
            continue
        iv = _implied_vol(m, S, K, T)
        if iv is None:
            continue
        x = math.log(max(1e-9, S / K))
        sp = max(1.0, spreads.get(sym, 10.0))
        xs.append(x)
        ys.append(iv)
        ws.append(1.0 / sp)
    if len(xs) < 6:
        return None

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    w = np.array(ws, dtype=float)
    X = np.column_stack([np.ones_like(x), x, x * x])

    def solve(weight_vec):
        W = np.diag(weight_vec)
        return np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)

    beta = solve(w)
    if robust:
        for _ in range(3):
            r = y - (X @ beta)
            med = float(np.median(r))
            mad = float(np.median(np.abs(r - med))) + 1e-9
            c = 2.5 * 1.4826 * mad
            rw = np.where(np.abs(r) <= c, 1.0, c / np.abs(r))
            beta = solve(w * rw)

    def pred(logm):
        return float(beta[0] + beta[1] * logm + beta[2] * logm * logm)

    return pred


def _bundle_theo_from_smile(S: float, T: float, pred):
    iv0 = max(0.05, min(1.5, pred(math.log(max(1e-9, S / 5000.0)))))
    iv1 = max(0.05, min(1.5, pred(math.log(max(1e-9, S / 5100.0)))))
    c0 = _bs_call(S, 5000.0, T, iv0)
    c1 = _bs_call(S, 5100.0, T, iv1)
    d0 = _bs_delta_call(S, 5000.0, T, iv0)
    d1 = _bs_delta_call(S, 5100.0, T, iv1)
    return c0 + c1, d0 + d1, 0.5 * (iv0 + iv1)


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
        d52 = state.order_depths.get(VEV_5200)
        d53 = state.order_depths.get(VEV_5300)

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

        sp0 = _spread(d0) or 99.0
        sp1 = _spread(d1) or 99.0
        ss = sp0 + sp1
        mids = {"VEV_5000": mid0, "VEV_5100": mid1}
        spreads = {"VEV_5000": sp0, "VEV_5100": sp1}
        for k in ("VEV_4000", "VEV_4500", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"):
            od = state.order_depths.get(k)
            if od is not None and od.buy_orders and od.sell_orders:
                mk = _mid(od)
                if mk is not None:
                    mids[k] = mk[0]
                    spreads[k] = _spread(od) or 10.0

        pred = _fit_smile_quadratic_logm(S, T, mids, spreads, robust=True)
        if pred is not None:
            theo_bundle, unit_straddle_delta, iv_ref = _bundle_theo_from_smile(S, T, pred)
            iv0 = max(0.05, min(1.5, pred(math.log(max(1e-9, S / 5000.0)))))
            iv1 = max(0.05, min(1.5, pred(math.log(max(1e-9, S / 5100.0)))))
        else:
            iv0 = _implied_vol(mid0, S, K0, T)
            iv1 = _implied_vol(mid1, S, K1, T)
            theo_bundle = _bs_call(S, K0, T, iv0) + _bs_call(S, K1, T, iv1)
            unit_straddle_delta = _bs_delta_call(S, K0, T, iv0) + _bs_delta_call(S, K1, T, iv1)
            iv_ref = 0.5 * (iv0 + iv1)
        bundle_mid = mid0 + mid1
        mis = bundle_mid - theo_bundle

        ve0 = _bs_vega(S, K0, T, iv0)
        ve1 = _bs_vega(S, K1, T, iv1)
        vega_ref = 0.5 * (ve0 + ve1)
        g0 = _bs_gamma(S, K0, T, iv0)
        g1 = _bs_gamma(S, K1, T, iv1)
        straddle_gamma = g0 + g1
        # Tight combined books: require larger absolute edge; wide books: slightly easier (per mis vs spread buckets).
        spread_adj = SPREAD_FLOOR_COEF * (SPREAD_SUM_REF - ss)
        mis_floor = (
            VEGA_MISPRICE_BASE
            + VEGA_MISPRICE_PER_VEGA * vega_ref
            + spread_adj
            + GAMMA_MISPRICE_PER_GAMMA * straddle_gamma
        )

        prev_s = td.get("_prev_S")
        dS = 0.0 if prev_s is None else (S - float(prev_s))
        td["_prev_S"] = S
        shock_edge = SHOCK_ABS_EDGE * (
            1.0
            + SHOCK_DS_SCALE * min(1.0, abs(dS) / SHOCK_DS_REF)
            + SHOCK_SPREAD_COEF * max(0.0, ss - SPREAD_SUM_REF)
        )

        shock = iv_ref - sigma_rv

        p0 = state.position.get(VEV_5000, 0)
        p1 = state.position.get(VEV_5100, 0)
        px = state.position.get(EXTRACT, 0)

        s52 = _spread(d52) if d52 is not None else None
        s53 = _spread(d53) if d53 is not None else None
        joint_5200_5300_tight = (
            s52 is not None
            and s53 is not None
            and s52 <= TIGHT_5200_5300_TH
            and s53 <= TIGHT_5200_5300_TH
        )
        clip = ORDER_CLIP_JOINT_TIGHT if joint_5200_5300_tight else ORDER_CLIP

        spread_ok = ss <= SPREAD_GATE_SUM

        wide_book = ss > SHOCK_WIDE_SS
        shock_edge_eff = shock_edge * (SHOCK_WIDE_MULT if wide_book else 1.0)
        shock_short = shock > shock_edge_eff
        shock_long = shock < -shock_edge_eff
        mis_short = mis > mis_floor
        mis_long = mis < -mis_floor
        # Shock-only entries are noisy in low-|dS| and wide-book regimes on tape; gate those specifically.
        allow_shock_only = abs(dS) >= SHOCK_ONLY_MIN_DS
        want_short = mis_short or (shock_short and (allow_shock_only or mis_short))
        want_long = mis_long or (shock_long and (allow_shock_only or mis_long))
        trade_short_bundle = want_short and (not want_long or mis > 0)
        trade_long_bundle = want_long and (not want_short or mis < 0)

        if spread_ok and trade_short_bundle:
            q = min(clip, POS_LIMITS[VEV_5000] + p0, POS_LIMITS[VEV_5100] + p1)
            q = max(0, q)
            if q > 0:
                out[VEV_5000].append(Order(VEV_5000, bb0, -q))
                out[VEV_5100].append(Order(VEV_5100, bb1, -q))
            hedge = int(round(unit_straddle_delta * q))
            if hedge > 0 and px < POS_LIMITS[EXTRACT]:
                hq = min(hedge, POS_LIMITS[EXTRACT] - px, clip * 3)
                if hq > 0:
                    out[EXTRACT].append(Order(EXTRACT, int(math.ceil(S)), hq))
            elif hedge < 0 and px > -POS_LIMITS[EXTRACT]:
                hq = min(-hedge, POS_LIMITS[EXTRACT] + px, clip * 3)
                if hq > 0:
                    out[EXTRACT].append(Order(EXTRACT, int(math.floor(S)), -hq))

        elif spread_ok and trade_long_bundle:
            q = min(clip, POS_LIMITS[VEV_5000] - p0, POS_LIMITS[VEV_5100] - p1)
            q = max(0, q)
            if q > 0:
                out[VEV_5000].append(Order(VEV_5000, ba0, q))
                out[VEV_5100].append(Order(VEV_5100, ba1, q))
            hedge = int(round(unit_straddle_delta * q))
            if hedge > 0 and px > -POS_LIMITS[EXTRACT]:
                hq = min(hedge, POS_LIMITS[EXTRACT] + px, clip * 3)
                if hq > 0:
                    out[EXTRACT].append(Order(EXTRACT, int(math.floor(S)), -hq))
            elif hedge < 0 and px < POS_LIMITS[EXTRACT]:
                hq = min(-hedge, POS_LIMITS[EXTRACT] - px, clip * 3)
                if hq > 0:
                    out[EXTRACT].append(Order(EXTRACT, int(math.ceil(S)), hq))

        return out, 0, json.dumps(td)
