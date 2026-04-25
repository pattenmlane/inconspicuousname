"""
Round 3 — cross-strike IV rank fade (r3v_cross_strike_iv_rank_07), iteration v19.

Thesis: rank **IV smile residuals** (not raw IV), then fade top/bottom decile with a neighbor
cap on **residual** gaps. Implied vol is solved from mids; a quadratic smile in
m_t = log(K/S) / sqrt(T) is fit each tick; residual_i = IV_i - q(m_t_i) (T in years, r=0),
matching the Frankfurt/IV-smile parameterization in round3 plotting helpers.

Timing / TTE: round3work/round3description.txt (CSV day 0/1/2 -> DTE 8/7/6 at open) and
intraday winding as round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3
(dte_eff = max(8 - csv_day - (timestamp//100)/10000, 1e-6); T = dte_eff/365).

Distinct family13-adjacent variant: underlying-propagation-aware residual rank/width by strike beta+delta buckets; regime signal uses EWMA RV minus IV spread (beta-weighted core vs wing ATM proxy) with cooldown.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState
from scipy.stats import norm

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]

NEIGHBOR_RESIDUAL_CAP = 0.12
BASE_Q = 8
MAX_CLUSTER_Q = 22
WARMUP_STEPS = 8
ACTION_STRIDE = 1
EMA_S_N = 12
_EMA_KEY = "ema_S"

# Regime on propagation-aware IV spread (core vs wings) against EWMA RV
RV_ALPHA = 0.06
REGIME_THRESHOLD = 0.03
COOLDOWN_TICKS = 20


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _sym_for_product(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _best_ba(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if depth is None:
        return None, None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None, None
    return max(buys.keys()), min(sells.keys())


def dte_effective(csv_day: int, timestamp: int) -> float:
    d0 = 8 - int(csv_day)
    prog = (int(timestamp) // 100) / 10_000.0
    return max(float(d0) - prog, 1e-6)


def t_years(csv_day: int, timestamp: int) -> float:
    return dte_effective(csv_day, timestamp) / 365.0


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def implied_vol_call(market: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-9:
        return float("nan")
    if market >= S - 1e-9:
        return float("nan")
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    lo, hi = 1e-5, 15.0
    fl, fh = f(lo), f(hi)
    if fl > 0 or fh < 0:
        return float("nan")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _update_csv_day(td: dict[str, Any], ts: int, S: float) -> int:
    if ts != 0:
        return int(td.get("csv_day", 0))
    hist = td.get("open_S_hist")
    if not isinstance(hist, list):
        hist = []
    cur = round(float(S), 2)
    if not hist or abs(float(hist[-1]) - cur) > 0.25:
        hist.append(cur)
    td["open_S_hist"] = hist[:4]
    return max(0, min(len(hist) - 1, 2))


def _ema(prev: float | None, x: float, n: int) -> float:
    if prev is None:
        return x
    a = 2.0 / (n + 1.0)
    return a * x + (1.0 - a) * prev




# Precomputed from underlying_propagation_iv_greeks analysis (step-5 sample)
_BETA_BY_STRIKE = {
    4000: 0.8147904803620967, 4500: 0.8863297384121156, 5000: 0.8698431297109179,
    5100: 0.7624399615754083, 5200: 0.5795214391756178, 5300: 0.36409920530957995,
    5400: 0.17095450179023666, 5500: 0.0749890839228015, 6000: 0.0, 6500: 0.0,
}


def _regime_params(regime: str) -> tuple[int, int, int]:
    # (width add in ticks, inventory cap, size multiplier %)
    if regime == "high":
        return 1, 240, 90
    if regime == "low":
        return 0, 280, 115
    return 1, 260, 100

def _quadratic_residuals(S: float, T: float, k_list: list[float], iv_list: list[float]) -> list[float] | None:
    """IV minus quadratic smile in m_t = log(K/S)/sqrt(T), np.polyval order a*m^2+b*m+c."""
    if len(iv_list) < 5 or T <= 0.0 or S <= 0.0:
        return None
    sqrtT = math.sqrt(T)
    m_arr = []
    y_arr = []
    for k, sig in zip(k_list, iv_list):
        if not math.isfinite(sig) or k <= 0.0:
            return None
        m_arr.append(math.log(k / S) / sqrtT)
        y_arr.append(sig)
    m_np = np.asarray(m_arr, dtype=float)
    y_np = np.asarray(y_arr, dtype=float)
    try:
        coeff = np.polyfit(m_np, y_np, 2)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(coeff)):
        return None
    pred = np.polyval(coeff, m_np)
    return (y_np - pred).tolist()


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        sym_u = _sym_for_product(state, "VELVETFRUIT_EXTRACT")
        if sym_u is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        du = depths.get(sym_u)
        ubb, uba = _best_ba(du)
        if ubb is None or uba is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        S_raw = 0.5 * (ubb + uba)
        ema_s = td.get(_EMA_KEY)
        ema_s_f = float(ema_s) if isinstance(ema_s, (int, float)) else None
        ema_s_f = _ema(ema_s_f, S_raw, EMA_S_N)
        td[_EMA_KEY] = ema_s_f
        S = ema_s_f

        csv_day = _update_csv_day(td, ts, S_raw)
        td["csv_day"] = csv_day

        if ts // 100 < WARMUP_STEPS:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        if (ts // 100) % ACTION_STRIDE != 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        T = t_years(csv_day, ts)

        ivs: list[float] = []
        k_list: list[float] = []
        bids: list[int] = []
        asks: list[int] = []
        syms: list[str] = []

        for prod in VOUCHERS:
            sym = _sym_for_product(state, prod)
            if sym is None:
                continue
            d = depths.get(sym)
            bb, ba = _best_ba(d)
            if bb is None or ba is None:
                continue
            mid = 0.5 * (bb + ba)
            K = float(prod.split("_")[1])
            iv = implied_vol_call(mid, S, K, T, 0.0)
            if not math.isfinite(iv):
                continue
            ivs.append(iv)
            k_list.append(K)
            bids.append(int(bb))
            asks.append(int(ba))
            syms.append(sym)

        n = len(ivs)
        if n < 8:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        # Propagation-aware regime: compare beta-weighted core IV to wing ATM proxy vs EWMA RV
        prev_s = td.get("prev_S")
        rv_ewma = float(td.get("rv_ewma", 0.0))
        if isinstance(prev_s, (int, float)) and float(prev_s) > 0:
            ret = math.log(max(S, 1e-6) / max(float(prev_s), 1e-6))
            rv_ewma = (1.0 - RV_ALPHA) * rv_ewma + RV_ALPHA * (ret * ret * 365.0 * 10_000.0)
        td["prev_S"] = S
        td["rv_ewma"] = rv_ewma

        core_num = 0.0
        core_den = 0.0
        wing_vals = []
        for i in range(n):
            k = int(round(k_list[i]))
            b = _BETA_BY_STRIKE.get(k, 0.0)
            if 5100 <= k <= 5400 and b > 0:
                core_num += b * ivs[i]
                core_den += b
            if k in (4000, 4500, 5500, 6000, 6500):
                wing_vals.append(ivs[i])
        core_iv = core_num / core_den if core_den > 1e-9 else float(sum(ivs) / len(ivs))
        wing_iv = float(sum(wing_vals) / len(wing_vals)) if wing_vals else core_iv
        spread_iv = core_iv - wing_iv
        rv_sigma = math.sqrt(max(rv_ewma, 0.0)) if rv_ewma > 0 else 0.0
        raw_sig = spread_iv - rv_sigma

        target_regime = "mid"
        if raw_sig > REGIME_THRESHOLD:
            target_regime = "high"
        elif raw_sig < -REGIME_THRESHOLD:
            target_regime = "low"

        regime = str(td.get("regime", "mid"))
        cd = int(td.get("regime_cd", 0))
        switched = 0
        if cd > 0:
            cd -= 1
        elif target_regime != regime:
            regime = target_regime
            cd = COOLDOWN_TICKS
            switched = 1
        td["regime"] = regime
        td["regime_cd"] = cd
        td["regime_switch"] = switched
        td["regime_signal"] = raw_sig

        res = _quadratic_residuals(S, T, k_list, ivs)
        if res is None or len(res) != n:
            rank_key = list(ivs)
        else:
            # propagation-aware residual scaling: emphasize strikes with real dP/dS coupling
            rank_key = []
            for i in range(n):
                k = int(round(k_list[i]))
                b = _BETA_BY_STRIKE.get(k, 0.0)
                rank_key.append(res[i] * (0.5 + b))

        order_idx = sorted(range(n), key=lambda i: rank_key[i])
        low_pair = set(order_idx[:2])
        high_pair = set(order_idx[-2:])

        def neighbor_rank_gap(i: int) -> float:
            gaps = []
            if i > 0:
                gaps.append(abs(rank_key[i] - rank_key[i - 1]))
            if i < n - 1:
                gaps.append(abs(rank_key[i] - rank_key[i + 1]))
            return max(gaps) if gaps else 0.0

        width_off, inv_cap, size_mult = _regime_params(td.get("regime","mid"))
        orders_by_sym: dict[str, list[Order]] = {}

        def add_o(sym: str, price: int, qty: int) -> None:
            if qty == 0:
                return
            orders_by_sym.setdefault(sym, []).append(Order(sym, int(price), int(qty)))

        for i in range(n):
            sym = syms[i]
            pos_i = int(pos.get(sym, 0))
            lim = min(300, inv_cap)
            bb2, ba2 = bids[i], asks[i]
            if neighbor_rank_gap(i) > NEIGHBOR_RESIDUAL_CAP:
                continue
            q_use = max(1, int(round(BASE_Q * (size_mult / 100.0))))
            if i > 0 and ((i - 1) in low_pair or (i - 1) in high_pair):
                q_use = min(MAX_CLUSTER_Q, BASE_Q * 2)
            if i < n - 1 and ((i + 1) in low_pair or (i + 1) in high_pair):
                q_use = min(MAX_CLUSTER_Q, max(q_use, BASE_Q * 2))

            if i in low_pair:
                qb = min(q_use, lim - pos_i)
                pxb = bb2 + width_off
                if qb > 0 and pxb < ba2:
                    add_o(sym, pxb, qb)
            if i in high_pair:
                qs = min(q_use, lim + pos_i)
                pxs = ba2 - width_off
                if qs > 0 and pxs > bb2:
                    add_o(sym, pxs, -qs)

        return orders_by_sym, 0, json.dumps(td, separators=(",", ":"))
