"""
Round 3 VEV (v8): same as **trader_v7** with **GAMMA_EDGE_SCALE=160** (grid vs v7’s 100).

TTE mapping (round3work/round3description.txt): tape file `prices_round_3_day_{d}.csv`
uses CSV `day` column = d. TTE in days = 8 - d (e.g. d=0 -> 8d, d=1 -> 7d, d=2 -> 6d).
Backtester sets PROSPERITY4_BACKTEST_DAY to the simulated day index so this matches the tape.
"""
from __future__ import annotations

import json
import math
import os
from datamodel import Order, OrderDepth, TradingState
import numpy as np
from scipy.interpolate import CubicSpline

R = 0.0
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI

VEV_SYMS = [
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
STRIKES = [int(s.split("_")[1]) for s in VEV_SYMS]
HYDRO = "HYDROGEL_PACK"
EXTRACT = "VELVETFRUIT_EXTRACT"
TRADEABLE = [HYDRO, EXTRACT] + VEV_SYMS

LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{s: 300 for s in VEV_SYMS},
}


def _tape_day() -> int:
    env = os.environ.get("PROSPERITY4_BACKTEST_DAY")
    if env is not None and env.lstrip("-").isdigit():
        return int(env)
    return 0


def tte_years_from_day(day_idx: int) -> float:
    dte = 8 - int(day_idx)
    return max(dte, 1) / 365.25


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * math.exp(-R * T) * _norm_cdf(d2)


def bs_delta_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    return float(_norm_cdf(d1))


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    return S * math.sqrt(T) * _norm_pdf(d1)


def bs_gamma(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sigma * sigma) * T) / v
    return _norm_pdf(d1) / (S * v)


def implied_vol(
    S: float, K: float, T: float, price: float, initial: float | None = None
) -> float | None:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if price < intrinsic - 1e-9:
        return None
    if bs_call(S, K, T, 4.5) < price - 1e-9:
        return None
    sigma = 0.28 if initial is None else max(1e-4, min(float(initial), 4.5))
    for _ in range(8):
        th = bs_call(S, K, T, sigma) - price
        if abs(th) < 1e-7:
            return sigma
        vg = bs_vega(S, K, T, sigma)
        if vg < 1e-14:
            break
        sigma -= th / vg
        sigma = max(1e-6, min(sigma, 4.5))
    # Bisection fallback (deep OTM / tiny vega)
    lo, hi = 1e-5, 4.5
    if bs_call(S, K, T, lo) > price:
        return None
    if bs_call(S, K, T, hi) < price:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv, av = depth.buy_orders[bb], -depth.sell_orders[ba]
    tot = bv + av
    if tot <= 0:
        return 0.5 * (bb + ba)
    return (bb * av + ba * bv) / tot


def fit_spline_ivs(logks: np.ndarray, ivs: np.ndarray, robust_iv_range: float) -> CubicSpline | None:
    """Return cubic spline IV(log K); optionally drop one outlier IV."""
    if len(logks) < 4:
        return None
    order = np.argsort(logks)
    x = logks[order].astype(float)
    y = ivs[order].astype(float)
    if float(np.max(y) - np.min(y)) > robust_iv_range and len(y) >= 6:
        med = float(np.median(y))
        drop = int(np.argmax(np.abs(y - med)))
        x = np.delete(x, drop)
        y = np.delete(y, drop)
    if len(x) < 4:
        return None
    return CubicSpline(x, y, bc_type="natural")


class Trader:
    # Edge in **same units as option prices** (seashells, integer book)
    TAKE_EDGE = 2.0
    MAKE_EDGE = 1.0
    MM_SIZE = 18
    TAKE_SIZE = 22
    # Scale for vega (per seashell) / spread (ticks) — small grid: v5=0.15, v6=0.25
    VEGA_EDGE_SCALE = 0.15
    # BS γ is O(1e-3) near ATM; map to small edge add in seashells (grid: v7=100, v8=160)
    GAMMA_EDGE_SCALE = 160.0
    ROBUST_IV_RANGE = 0.35
    # Micro fair on extract (Kalman-ish)
    EX_K = 0.12

    def run(self, state: TradingState):
        td: dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except json.JSONDecodeError:
                td = {}

        day_idx = _tape_day()
        T = tte_years_from_day(day_idx)

        result: dict[str, list[Order]] = {p: [] for p in TRADEABLE}

        ex = state.order_depths.get(EXTRACT)
        if ex is None or not ex.buy_orders or not ex.sell_orders:
            return result, 0, json.dumps(td)

        S = wall_mid(ex)
        if S is None or S <= 0:
            return result, 0, json.dumps(td)

        f_ex = td.get("_fex")
        if f_ex is None:
            f_ex = S
        else:
            f_ex = float(f_ex) + self.EX_K * (S - float(f_ex))
        td["_fex"] = f_ex

        logks: list[float] = []
        ivs: list[float] = []
        mids: dict[str, float] = {}

        iv_prev = td.get("_iv")
        if not isinstance(iv_prev, dict):
            iv_prev = {}

        s_opt = float(S)
        for sym, K in zip(VEV_SYMS, STRIKES):
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            wm = wall_mid(d)
            if wm is None:
                continue
            p0 = iv_prev.get(sym)
            init = float(p0) if isinstance(p0, (int, float)) else None
            iv = implied_vol(s_opt, float(K), T, float(wm), initial=init)
            if iv is None:
                continue
            iv_prev[sym] = iv
            logks.append(math.log(float(K)))
            ivs.append(iv)
            mids[sym] = float(wm)
        td["_iv"] = iv_prev

        if len(logks) < 4:
            self._append_extract_hydro(result, ex, state, float(f_ex))
            td["_fex"] = f_ex
            return result, 0, json.dumps(td)

        cs = fit_spline_ivs(np.array(logks), np.array(ivs), self.ROBUST_IV_RANGE)
        if cs is None:
            self._append_extract_hydro(result, ex, state, float(f_ex))
            td["_fex"] = f_ex
            return result, 0, json.dumps(td)

        # Skew extract quotes by net BS delta of voucher inventory (vega-free lean)
        net_delta = 0.0
        for sym, K in zip(VEV_SYMS, STRIKES):
            pos = state.position.get(sym, 0)
            if pos == 0:
                continue
            sig = float(cs(math.log(float(K))))
            net_delta += pos * bs_delta_call(s_opt, float(K), T, sig)

        result[EXTRACT].extend(
            self._quote_extract(ex, state.position.get(EXTRACT, 0), float(f_ex), delta_skew=net_delta)
        )

        for sym, K in zip(VEV_SYMS, STRIKES):
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            if sym not in mids:
                continue
            fair_iv = float(cs(math.log(float(K))))
            fair = bs_call(s_opt, float(K), T, fair_iv)
            sig = max(fair_iv, 1e-6)
            vega = bs_vega(s_opt, float(K), T, sig)
            gamm = bs_gamma(s_opt, float(K), T, sig)
            pos = state.position.get(sym, 0)
            lim = LIMITS[sym]
            result[sym].extend(
                self._vev_orders(
                    sym, d, pos, lim, fair, self.TAKE_EDGE, self.MAKE_EDGE, self.MM_SIZE, self.TAKE_SIZE, vega, gamm
                )
            )

        hd = state.order_depths.get(HYDRO)
        if hd is not None and hd.buy_orders and hd.sell_orders:
            result[HYDRO].extend(self._quote_hydro(hd, state.position.get(HYDRO, 0)))
        td["_fex"] = f_ex
        return result, 0, json.dumps(td)

    def _append_extract_hydro(self, result: dict, ex: OrderDepth, state: TradingState, f_ex: float) -> None:
        result[EXTRACT].extend(self._quote_extract(ex, state.position.get(EXTRACT, 0), f_ex))
        hd = state.order_depths.get(HYDRO)
        if hd is not None and hd.buy_orders and hd.sell_orders:
            result[HYDRO].extend(self._quote_hydro(hd, state.position.get(HYDRO, 0)))

    def _vev_orders(
        self,
        sym: str,
        depth: OrderDepth,
        pos: int,
        lim: int,
        fair: float,
        take_edge: float,
        make_edge: float,
        mm_size: int,
        take_size: int,
        vega: float,
        gamma: float,
    ) -> list[Order]:
        orders: list[Order] = []
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        best_bid = int(bb)
        best_ask = int(ba)
        spr = max(0.0, float(best_ask - best_bid))
        den = 1.0 + spr / 8.0
        # Tighten when vega is large and spread is narrow; loosen when book is wide.
        vega_adj = self.VEGA_EDGE_SCALE * abs(vega) / den
        # High |gamma| → fair is sensitive to S noise; be less aggressive
        g_adj = self.GAMMA_EDGE_SCALE * abs(gamma) / den
        take_e = max(0.5, take_edge - vega_adj + g_adj)
        make_e = max(0.0, make_edge - 0.4 * vega_adj + 0.25 * g_adj)

        # Integer book: compare in float fair space, then clip prices
        # Takes
        if best_ask <= fair - take_e + 1e-9 and pos < lim:
            q = min(take_size, lim - pos)
            if q > 0:
                orders.append(Order(sym, best_ask, q))
        if best_bid >= fair + take_e - 1e-9 and pos > -lim:
            q = min(take_size, lim + pos)
            if q > 0:
                orders.append(Order(sym, best_bid, -q))

        bid_anchor = int(math.floor(fair - make_e))
        ask_anchor = int(math.ceil(fair + make_e))
        # Quotes (passive); bid 0 is valid for penny options (see tape)
        bid_p = min(best_bid + 1, bid_anchor)
        bid_p = max(0, bid_p)
        if bid_p < best_ask and pos < lim:
            q = min(mm_size, lim - pos)
            if q > 0:
                orders.append(Order(sym, bid_p, q))
        ask_p = max(best_ask - 1, ask_anchor)
        if ask_p > best_bid and pos > -lim:
            q = min(mm_size, lim + pos)
            if q > 0:
                orders.append(Order(sym, ask_p, -q))
        return orders

    def _quote_extract(self, depth: OrderDepth, pos: int, fair: float, delta_skew: float = 0.0) -> list[Order]:
        orders: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return orders
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        skew = int(round(max(-3.0, min(3.0, 0.02 * delta_skew))))
        fi = int(round(fair)) + skew
        edge = 2
        lim = LIMITS[EXTRACT]
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < lim:
            orders.append(Order(EXTRACT, bid_p, min(25, lim - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -lim:
            orders.append(Order(EXTRACT, ask_p, -min(25, lim + pos)))
        return orders

    def _quote_hydro(self, depth: OrderDepth, pos: int) -> list[Order]:
        orders: list[Order] = []
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        lim = LIMITS[HYDRO]
        mid = 0.5 * (bb + ba)
        fi = int(round(mid))
        edge = 3
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < lim:
            orders.append(Order(HYDRO, bid_p, min(20, lim - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -lim:
            orders.append(Order(HYDRO, ask_p, -min(20, lim + pos)))
        return orders

