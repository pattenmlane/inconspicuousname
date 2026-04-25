"""
Round 3 VEV (v19): v11 + top-of-book depth scaling for MM/TAKE sizes (BETA=0.5).
See v20 for a milder grid (BETA=0.25).

Tape analysis (vev_book_depth_by_strike.csv): pooled mean min(bid1_vol, ask1_vol) per row is ~16
across all VEV; deep ITM (4000/4500) have thinner tops (~9–11) than ATM/OTM (5200+ ~20–22).
Scale size by (min(REF, min_top) / REF)^BETA after the v11 delta taper so we quote smaller
when the visible book is thin, and at most baseline when it is at or above reference.

TTE: `round3work/round3description.txt` — tape day d, TTE_days = 8 - d, T = TTE_days/365.25.
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


def top_min_bid_ask_liquidity(depth: OrderDepth) -> float:
    """Positive volumes at best bid and ask; return min of the two (tape-consistent with analysis)."""
    if not depth.buy_orders or not depth.sell_orders:
        return 0.0
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv = float(max(0, depth.buy_orders[bb]))
    av = float(max(0, -depth.sell_orders[ba]))
    return float(min(bv, av))


def robust_spline_nodes(
    logks: np.ndarray, ivs: np.ndarray, robust_iv_range: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return sorted (x, y) IV nodes, optional global outlier drop (same as v5)."""
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
    return x, y


def loo_spline_iv_at(x: np.ndarray, y: np.ndarray, j: int, logk: float) -> float:
    """IV at logk from natural cubic spline that omits point j; fallback if <5 nodes."""
    if len(x) >= 5:
        xl = np.delete(x, j)
        yl = np.delete(y, j)
        if len(xl) >= 4:
            cs = CubicSpline(xl, yl, bc_type="natural")
            return float(cs(float(logk)))
    cs = CubicSpline(x, y, bc_type="natural")
    return float(cs(float(logk)))


def index_of_logk(x: np.ndarray, logk: float) -> int | None:
    """Node index for this log(strike), or None if this strike was not in the robust fit (outlier row)."""
    for j in range(len(x)):
        if abs(float(x[j]) - logk) < 1e-5:
            return j
    return None


def fair_iv_for_strike(x: np.ndarray, y: np.ndarray, logk: float) -> float:
    j = index_of_logk(x, logk)
    if j is None or len(x) < 4:
        cs = CubicSpline(x, y, bc_type="natural")
        return float(cs(float(logk)))
    if len(x) < 5:
        cs = CubicSpline(x, y, bc_type="natural")
        return float(cs(float(logk)))
    return loo_spline_iv_at(x, y, j, logk)


class Trader:
    TAKE_EDGE = 2.0
    MAKE_EDGE = 1.0
    MM_SIZE = 18
    TAKE_SIZE = 22
    VEGA_EDGE_SCALE = 0.15
    ROBUST_IV_RANGE = 0.35
    EX_K = 0.12
    DELTA_HEADROOM = 0.50
    # Pooled-tape ref (mean min top depth across VEV rows); see analyze_vev_book_depth_by_strike.py
    DEPTH_REF = 16.0
    DEPTH_BETA = 0.5

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

        nodes = robust_spline_nodes(np.array(logks), np.array(ivs), self.ROBUST_IV_RANGE)
        if nodes is None:
            self._append_extract_hydro(result, ex, state, float(f_ex))
            td["_fex"] = f_ex
            return result, 0, json.dumps(td)
        x, y = nodes

        net_delta = 0.0
        for sym, K in zip(VEV_SYMS, STRIKES):
            pos = state.position.get(sym, 0)
            if pos == 0:
                continue
            if sym not in mids:
                continue
            lk = math.log(float(K))
            sig = fair_iv_for_strike(x, y, lk)
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
            lk = math.log(float(K))
            fair_iv = fair_iv_for_strike(x, y, lk)
            fair = bs_call(s_opt, float(K), T, fair_iv)
            vega = bs_vega(s_opt, float(K), T, max(fair_iv, 1e-6))
            pos = state.position.get(sym, 0)
            lim = LIMITS[sym]
            delta = bs_delta_call(s_opt, float(K), T, max(fair_iv, 1e-6))
            result[sym].extend(
                self._vev_orders(
                    sym, d, pos, lim, fair, self.TAKE_EDGE, self.MAKE_EDGE, self.MM_SIZE, self.TAKE_SIZE, vega, delta
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
        delta: float,
    ) -> list[Order]:
        orders: list[Order] = []
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        best_bid = int(bb)
        best_ask = int(ba)
        spr = max(0.0, float(best_ask - best_bid))
        adj = self.VEGA_EDGE_SCALE * abs(vega) / (1.0 + spr / 8.0)
        take_e = max(0.5, take_edge - adj)
        make_e = max(0.0, make_edge - 0.4 * adj)

        # Delta-cap-aware size: if option contributes large delta-per-lot, reduce size near limits.
        abs_d = max(0.02, abs(delta))
        delta_util = abs(pos) * abs_d / max(1.0, lim * abs_d)
        # Keep VEV_4000 a bit more aggressive; other strikes taper faster.
        taper = 1.0 - max(0.0, delta_util - self.DELTA_HEADROOM)
        if sym != "VEV_4000":
            taper *= 0.8
        taper = max(0.25, min(1.0, taper))
        mm_size = max(6, int(round(mm_size * taper)))
        take_size = max(8, int(round(take_size * taper)))

        # Top-of-book depth: downscale when visible liquidity is below pooled tape ref (~16)
        mtop = top_min_bid_ask_liquidity(depth)
        ref = max(1.0, self.DEPTH_REF)
        capped = min(ref, max(1.0, mtop))
        dscale = (capped / ref) ** self.DEPTH_BETA
        dscale = max(0.35, min(1.0, dscale))
        mm_size = max(6, int(round(mm_size * dscale)))
        take_size = max(8, int(round(take_size * dscale)))

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
