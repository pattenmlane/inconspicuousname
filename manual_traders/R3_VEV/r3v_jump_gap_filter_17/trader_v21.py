"""
v21: **vouchers_final_strategy** (STRATEGY.txt / ORIGINAL_DISCORD_QUOTES) — joint 5200+5300 TOB spread
gate only; no legacy v17 jump/PTP pause or v15 move-asymmetry (removed per deprioritized launch thesis).

Tight = both VEV_5200 and VEV_5300 have ask_1 - bid_1 <= 2. **Soft gate:** VEVs always trade;
when tight use full TAKE/MAKE/MMS/TAKE_SIZ; when wide scale down sizes and **widen** edges (wide-book
regime: Sonic — execution noise dominates).

Family-12 LOO spline fair kept for VEV **relative** value only.
**No HYDROGEL_PACK**. TTE: t_years_effective; PROSPERITY4_BACKTEST_DAY for day index.
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
EXTRACT = "VELVETFRUIT_EXTRACT"
TRADEABLE = [EXTRACT] + VEV_SYMS
SY_5200 = "VEV_5200"
SY_5300 = "VEV_5300"
# round3work/vouchers_final_strategy/STRATEGY.txt; outputs/r3_tight_spread_summary.txt
_TIGHT_TOB = 2

LIMITS: dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 200,
    **{s: 300 for s in VEV_SYMS},
}

_REFIT_CADENCE = 40
_TD_KEY = "r3v_spline21"

# Soft gate: when joint tight is OFF, widen these edges and halve (rounded) sizes
_WIDE_TAKE_BUMP = 1.0
_WIDE_MAKE_BUMP = 0.5
_WIDE_SIZE_FRAC = 0.5


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


def tob_spread(depth: OrderDepth) -> int | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    return int(ba) - int(bb)


def joint_tight_gate_5200_5300(state: TradingState) -> bool:
    d52 = state.order_depths.get(SY_5200)
    d53 = state.order_depths.get(SY_5300)
    if d52 is None or d53 is None or not d52.buy_orders or not d52.sell_orders:
        return False
    if not d53.buy_orders or not d53.sell_orders:
        return False
    s52, s53 = tob_spread(d52), tob_spread(d53)
    if s52 is None or s53 is None:
        return False
    return s52 <= _TIGHT_TOB and s53 <= _TIGHT_TOB


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
        bu["prev_S"] = s_opt

        last_refit = int(bu.get("last_refit_tick", -10**9))
        need_refit = (bu.get("sx") is None) or (tick - last_refit >= _REFIT_CADENCE)

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
            self._append_extract_only(out, ex, state, float(f_ex), joint_tight=joint_tight_gate_5200_5300(state))
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
            self._append_extract_only(out, ex, state, float(f_ex), joint_tight=joint_tight_gate_5200_5300(state))
            return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)

        joint_t = joint_tight_gate_5200_5300(state)

        net_delta = 0.0
        for sym, K in zip(VEV_SYMS, STRIKES):
            pos = state.position.get(sym, 0)
            if pos == 0:
                continue
            lk = math.log(float(K))
            sig = fair_iv_for_strike(xa, ya, lk)
            net_delta += int(pos) * bs_delta_call(s_opt, float(K), T, float(sig))

        ex_pos = int(state.position.get(EXTRACT, 0) or 0)
        if joint_t:
            ex_clip = 25
        else:
            ex_clip = 10
        out[EXTRACT].extend(
            self._quote_extract(
                ex, ex_pos, float(f_ex), delta_skew=net_delta, max_q_per_side=ex_clip
            )
        )

        if joint_t:
            te, me, mms, tks = self.TAKE_EDGE, self.MAKE_EDGE, self.MM_SIZE, self.TAKE_SIZE
        else:
            te = self.TAKE_EDGE + _WIDE_TAKE_BUMP
            me = self.MAKE_EDGE + _WIDE_MAKE_BUMP
            mms = max(1, int(round(self.MM_SIZE * _WIDE_SIZE_FRAC)))
            tks = max(1, int(round(self.TAKE_SIZE * _WIDE_SIZE_FRAC)))
        for sym, K in zip(VEV_SYMS, STRIKES):
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            if sym not in mids:
                continue
            lk = math.log(float(K))
            fair_iv = fair_iv_for_strike(xa, ya, lk)
            fair = bs_call(s_opt, float(K), T, float(fair_iv))
            pos = state.position.get(sym, 0) or 0
            lim = LIMITS[sym]
            out[sym].extend(
                self._vev_orders(sym, d, int(pos), lim, fair, te, me, mms, tks)
            )

        return out, 0, json.dumps({_TD_KEY: bu}, separators=(",", ":"))

    def _append_extract_only(
        self,
        out: dict,
        ex: OrderDepth,
        state: TradingState,
        f_ex: float,
        *,
        joint_tight: bool,
    ) -> None:
        ex_clip = 25 if joint_tight else 10
        out[EXTRACT].extend(
            self._quote_extract(
                ex, int(state.position.get(EXTRACT, 0) or 0), f_ex, max_q_per_side=ex_clip
            )
        )

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

    def _quote_extract(
        self,
        depth: OrderDepth,
        pos: int,
        fair: float,
        delta_skew: float = 0.0,
        *,
        max_q_per_side: int = 25,
    ) -> list[Order]:
        olist: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return olist
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        skew = int(round(max(-3.0, min(3.0, 0.02 * delta_skew))))
        fi = int(round(fair)) + skew
        edge = 2
        limu = LIMITS[EXTRACT]
        mq = max(1, min(int(max_q_per_side), 25))
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < limu:
            olist.append(Order(EXTRACT, bid_p, min(mq, limu - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -limu:
            olist.append(Order(EXTRACT, ask_p, -min(mq, limu + pos)))
        return olist
