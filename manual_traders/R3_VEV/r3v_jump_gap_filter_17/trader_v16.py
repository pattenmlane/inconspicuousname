"""
Family-12: same core as v13 (LOO spline fair, intraday T, jump/cadence refit).

v16 (iteration 16): sign-asymmetric strike-aware fair tilt from tape
up–down *residual* asymmetry on *large* |ΔS| (pooled days 0–2, R3 tapes).

`analyze_underlying_updown_asymmetry.py` regresses dV on dS separately for
up and down; pooled betas are ~symmetric, but for |dS| ≥ p80 (~1.5) the
residual (dV - bpool*dS) is systematically **negative** on up moves and
**positive** on down moves (option mids lag a linear OLS).

Per-strike gap λ = mean_resid_up - mean_resid_down (e.g. ~0.11 at VEV_5000).
We shift BS fair by:

  fair_adj = fair - sign(ΔS) * w * λ,
  w = 0.45 * min(1, |dS|/3)  when |dS| ≥ 0.4,

so after **up** we quote **lower** fair (more likely to **lift** the ask, less
likely to get lifted on the bid); after **down** the opposite. This ties the
`jump / widen` thesis to a **signed** dislocation: large moves are the regime
where neighbor compression matters; we do not reintroduce v14's hard beta cut.

TTE: round3work/round3description.txt. PROSPERITY4_BACKTEST_DAY for day index only.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:  # backtester package
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"))
from plot_iv_smile_round3 import t_years_effective  # noqa: E402

R = 0.0
_SQRT2PI = math.sqrt(2.0 * math.pi)

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

_JUMP_DS = 3.0
_REFIT_CADENCE = 40
_TD_KEY = "r3v_spline16"

# From analysis_underlying_updown_asymmetry.json: per strike,
#   mean_resid_up - mean_resid_down  on |dS| >= p80 (large-move block).
# 6000/6500: no rows in that block use defaults from adjacent OTM.
_ASYMM_LAMBDA: dict[int, float] = {
    4000: 0.13343202292026972,
    4500: 0.08318145240195498,
    5000: 0.10702335504677515,
    5100: 0.09107549579043677,
    5200: 0.06910996335431449,
    5300: 0.045600445148320876,
    5400: 0.02267741029200571,
    5500: 0.011281385607655536,
    6000: 0.0,
    6500: 0.0,
}
_MOVE_EDGE_WMAX = 0.45
_MOVE_EDGE_TAU = 0.4

# From lead-lag analysis: down-shock follow-through stronger at these strikes;
# reduce/reverse down-side fair uplift there to avoid catching falling voucher mids.
_DOWN_MULT_BY_STRIKE: dict[int, float] = {
    4000: -0.25,
    5100: -0.45,
    5200: -0.45,
    5400: -0.35,
}
_DEFAULT_DOWN_MULT = 0.25
_UP_MULT = 1.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def _tape_day() -> int:
    env = os.environ.get("PROSPERITY4_BACKTEST_DAY")
    if env is not None and env.lstrip("-").isdigit():
        return int(env)
    return 0


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


def robust_spline_nodes(
    logks: np.ndarray, ivs: np.ndarray, robust_iv_range: float
) -> tuple[np.ndarray, np.ndarray] | None:
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
    if len(x) >= 5:
        xl = np.delete(x, j)
        yl = np.delete(y, j)
        if len(xl) >= 4:
            cs = CubicSpline(xl, yl, bc_type="natural")
            return float(cs(float(logk)))
    cs = CubicSpline(x, y, bc_type="natural")
    return float(cs(float(logk)))


def index_of_logk(x: np.ndarray, logk: float) -> int | None:
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


def fair_adjust_extract_asym(
    fair: float,
    K: int,
    dS_abs: float,
    sign_dS: float,
) -> float:
    lam = float(_ASYMM_LAMBDA.get(int(K), 0.0))
    if lam <= 0.0 or dS_abs < _MOVE_EDGE_TAU or abs(sign_dS) < 0.5:
        return float(fair)
    w = _MOVE_EDGE_WMAX * min(1.0, dS_abs / _JUMP_DS)
    if sign_dS > 0:
        mult = _UP_MULT
    else:
        mult = float(_DOWN_MULT_BY_STRIKE.get(int(K), _DEFAULT_DOWN_MULT))
    return float(fair - sign_dS * w * lam * mult)


class Trader:
    TAKE_EDGE = 2.0
    MAKE_EDGE = 1.0
    MM_SIZE = 18
    TAKE_SIZE = 22
    ROBUST_IV_RANGE = 0.35
    EX_K = 0.12

    def run(self, state: TradingState):
        bu: dict[str, Any] = {}
        if state.traderData:
            try:
                o = json.loads(state.traderData)
                if isinstance(o, dict) and _TD_KEY in o and isinstance(o[_TD_KEY], dict):
                    bu = o[_TD_KEY]
            except (json.JSONDecodeError, TypeError, KeyError):
                bu = {}

        day_idx = _tape_day()
        ts = int(getattr(state, "timestamp", 0))
        tick = ts // 100
        T = float(t_years_effective(int(day_idx), int(ts)))

        out: dict[str, list[Order]] = {p: [] for p in TRADEABLE}

        ex = state.order_depths.get(EXTRACT)
        if ex is None or not ex.buy_orders or not ex.sell_orders:
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        S = wall_mid(ex)
        if S is None or S <= 0:
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        s_opt = float(S)
        prev_s = bu.get("prev_S")
        if prev_s is None:
            sign_dS = 0.0
        else:
            ps = float(prev_s)
            if s_opt > ps + 1e-9:
                sign_dS = 1.0
            elif s_opt < ps - 1e-9:
                sign_dS = -1.0
            else:
                sign_dS = 0.0
        dS = 0.0 if prev_s is None else abs(s_opt - float(prev_s))
        bu["prev_S"] = s_opt

        last_refit = int(bu.get("last_refit_tick", -10**9))
        need_refit = (bu.get("sx") is None) or (tick - last_refit >= _REFIT_CADENCE) or (dS >= _JUMP_DS)

        f_ex = bu.get("_fex")
        if f_ex is None:
            f_ex = s_opt
        else:
            f_ex = float(f_ex) + self.EX_K * (s_opt - float(f_ex))
        bu["_fex"] = f_ex

        logks: list[float] = []
        ivs: list[float] = []
        mids: dict[str, float] = {}
        iv_prev = bu.get("_iv")
        if not isinstance(iv_prev, dict):
            iv_prev = {}
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
        bu["_iv"] = iv_prev

        if len(logks) < 4:
            self._append_extract_hydro(out, ex, state, float(f_ex))
            bu["prev_S"] = s_opt
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        if need_refit:
            rnodes = robust_spline_nodes(np.array(logks), np.array(ivs), self.ROBUST_IV_RANGE)
            if rnodes is not None:
                sx, sy = rnodes
                bu["sx"] = [float(t) for t in sx.tolist()]
                bu["sy"] = [float(t) for t in sy.tolist()]
                bu["last_refit_tick"] = tick
        x = bu.get("sx")
        y = bu.get("sy")
        if not isinstance(x, list) or not isinstance(y, list) or len(x) < 4:
            self._append_extract_hydro(out, ex, state, float(f_ex))
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)

        net_delta = 0.0
        for sym, K in zip(VEV_SYMS, STRIKES):
            pos = state.position.get(sym, 0)
            if pos == 0:
                continue
            lk = math.log(float(K))
            sig = fair_iv_for_strike(xa, ya, lk)
            net_delta += int(pos) * bs_delta_call(s_opt, float(K), T, float(sig))

        out[EXTRACT].extend(
            self._quote_extract(ex, state.position.get(EXTRACT, 0), float(f_ex), delta_skew=net_delta)
        )

        for sym, K in zip(VEV_SYMS, STRIKES):
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            if sym not in mids:
                continue
            lk = math.log(float(K))
            fair_iv = fair_iv_for_strike(xa, ya, lk)
            fair = bs_call(s_opt, float(K), T, float(fair_iv))
            Kint = int(K)
            fair_a = fair_adjust_extract_asym(fair, Kint, dS, sign_dS)
            pos = state.position.get(sym, 0) or 0
            lim = LIMITS[sym]
            out[sym].extend(
                self._vev_orders(
                    sym, d, int(pos), lim, fair_a, self.TAKE_EDGE, self.MAKE_EDGE, self.MM_SIZE, self.TAKE_SIZE
                )
            )

        hd = state.order_depths.get(HYDRO)
        if hd is not None and hd.buy_orders and hd.sell_orders:
            out[HYDRO].extend(self._quote_hydro(hd, int(state.position.get(HYDRO, 0) or 0)))
        return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

    def _append_extract_hydro(self, out: dict, ex: OrderDepth, state: TradingState, f_ex: float) -> None:
        out[EXTRACT].extend(self._quote_extract(ex, int(state.position.get(EXTRACT, 0) or 0), f_ex))
        hd = state.order_depths.get(HYDRO)
        if hd is not None and hd.buy_orders and hd.sell_orders:
            out[HYDRO].extend(self._quote_hydro(hd, int(state.position.get(HYDRO, 0) or 0)))

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
    ) -> list[Order]:
        olist: list[Order] = []
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        best_bid = int(bb)
        best_ask = int(ba)
        if best_ask <= fair - take_edge + 1e-9 and pos < lim:
            q = min(take_size, lim - pos)
            if q > 0:
                olist.append(Order(sym, best_ask, q))
        if best_bid >= fair + take_edge - 1e-9 and pos > -lim:
            q = min(take_size, lim + pos)
            if q > 0:
                olist.append(Order(sym, best_bid, -q))
        bid_anchor = int(math.floor(fair - make_edge))
        ask_anchor = int(math.ceil(fair + make_edge))
        bid_p = min(best_bid + 1, bid_anchor)
        bid_p = max(0, bid_p)
        if bid_p < best_ask and pos < lim:
            q = min(mm_size, lim - pos)
            if q > 0:
                olist.append(Order(sym, bid_p, q))
        ask_p = max(best_ask - 1, ask_anchor)
        if ask_p > best_bid and pos > -lim:
            q = min(mm_size, lim + pos)
            if q > 0:
                olist.append(Order(sym, ask_p, -q))
        return olist

    def _quote_extract(self, depth: OrderDepth, pos: int, fair: float, delta_skew: float = 0.0) -> list[Order]:
        olist: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return olist
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        skew = int(round(max(-3.0, min(3.0, 0.02 * delta_skew))))
        fi = int(round(fair)) + skew
        edge = 2
        limu = LIMITS[EXTRACT]
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < limu:
            olist.append(Order(EXTRACT, bid_p, min(25, limu - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -limu:
            olist.append(Order(EXTRACT, ask_p, -min(25, limu + pos)))
        return olist

    def _quote_hydro(self, depth: OrderDepth, pos: int) -> list[Order]:
        olist: list[Order] = []
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        limh = LIMITS[HYDRO]
        mid = 0.5 * (int(bb) + int(ba))
        fi = int(round(mid))
        edge = 3
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < limh:
            olist.append(Order(HYDRO, bid_p, min(20, limh - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -limh:
            olist.append(Order(HYDRO, ask_p, -min(20, limh + pos)))
        return olist
