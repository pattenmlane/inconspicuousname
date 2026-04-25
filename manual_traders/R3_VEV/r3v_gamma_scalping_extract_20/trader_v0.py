"""
Gamma-scalping extract (Round 3): long net options gamma near ATM; hedge on VELVETFRUIT_EXTRACT.

TTE for historical tapes (round3work/round3description.txt pattern): tape day index 0,1,2 maps to
TTE 8d, 7d, 6d (same offset as the spec example: historical day +1 -> 8d, etc.). We infer tape day
from timestamp resets across backtest days (timestamp goes back to 0).
"""
from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Any

from datamodel import Order, OrderDepth, TradingState

_N = NormalDist()

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV = [f"VEV_{k}" for k in STRIKES]
PRODUCTS = [HYDRO, EXTRACT] + VEV

LIMITS = {
    HYDRO: 200,
    EXTRACT: 200,
    **{v: 300 for v in VEV},
}

# TTE in whole days for tape day index 0,1,2 (see strategy doc / round3description)
TTE_BY_TAPE_DAY = (8, 7, 6)
DAYS_PER_YEAR = 365.0
R_RATE = 0.0

# --- strategy params (v0 baseline; sweep in v1) ---
GAMMA_THRESH = 0.00012
OPTION_SIZE = 14
HEDGE_BAND = 8
REQUOTE_EVERY = 3
MM_EDGE = 1
HYDRO_EDGE = 6
HYDRO_SIZE = 5


def _mid(depth: OrderDepth | None) -> float | None:
    if not depth or not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders.keys())
    ba = min(depth.sell_orders.keys())
    return (bb + ba) / 2.0


def _best(depth: OrderDepth | None) -> tuple[int | None, int | None]:
    if not depth:
        return None, None
    bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return bb, ba


def _cdf(x: float) -> float:
    return _N.cdf(x)


def _pdf(x: float) -> float:
    return _N.pdf(x)


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R_RATE + 0.5 * sigma * sigma) * T) / st
    d2 = d1 - st
    return S * _cdf(d1) - K * math.exp(-R_RATE * T) * _cdf(d2)


def bs_delta_gamma(S: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return ((1.0 if S > K else 0.0), 0.0)
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (R_RATE + 0.5 * sigma * sigma) * T) / st
    delta = _cdf(d1)
    gamma = _pdf(d1) / (S * st)
    return delta, gamma


def implied_vol(mid: float, S: float, K: float, T: float) -> float | None:
    if mid <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if mid + 1e-9 < intrinsic:
        return None
    lo, hi = 1e-5, 4.0
    for _ in range(55):
        sig = 0.5 * (lo + hi)
        p = bs_call_price(S, K, T, sig)
        if p > mid:
            hi = sig
        else:
            lo = sig
    return 0.5 * (lo + hi)


def atm_symbol(S: float) -> str:
    k = min(STRIKES, key=lambda x: abs(float(x) - S))
    return f"VEV_{k}"


class Trader:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        conversions = 0

        try:
            store: dict[str, Any] = json.loads(state.traderData) if (state.traderData or "").strip() else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        obs = getattr(state.observations, "plainValueObservations", None) or {}
        if "__BT_TAPE_DAY__" in obs:
            tape_day = int(obs["__BT_TAPE_DAY__"])
        else:
            last_ts = int(store.get("last_ts", -1))
            tape_day = int(store.get("tape_day", 0))
            if state.timestamp == 0 and last_ts > 50_000:
                tape_day = min(tape_day + 1, 2)
            store["tape_day"] = tape_day
            store["last_ts"] = int(state.timestamp)

        tte_days = TTE_BY_TAPE_DAY[tape_day]
        T = tte_days / DAYS_PER_YEAR

        d_ex = state.order_depths.get(EXTRACT)
        S = _mid(d_ex)
        if S is None:
            return result, conversions, json.dumps(store)

        sym = atm_symbol(S)
        K = float(sym.split("_")[1])
        d_opt = state.order_depths.get(sym)
        mid_c = _mid(d_opt)
        if mid_c is None:
            return result, conversions, json.dumps(store)

        iv = implied_vol(mid_c, S, K, T)
        if iv is None:
            return result, conversions, json.dumps(store)

        iv_ema = store.get("iv_ema")
        if iv_ema is None:
            iv_ema = iv
        else:
            iv_ema = 0.08 * iv + 0.92 * float(iv_ema)
        store["iv_ema"] = iv_ema

        delta, gamma = bs_delta_gamma(S, K, T, float(iv_ema))
        store["last_gamma"] = gamma
        store["last_delta"] = delta

        pos_ex = int(state.position.get(EXTRACT, 0))
        pos_opt = int(state.position.get(sym, 0))

        # Extract hedge: target short extract against long call delta
        opt_delta_exposure = pos_opt * delta
        target_ex = int(round(-opt_delta_exposure))
        target_ex = max(-LIMITS[EXTRACT], min(LIMITS[EXTRACT], target_ex))
        if pos_ex > target_ex + HEDGE_BAND:
            bb, ba = _best(d_ex)
            if bb is not None:
                q = min(pos_ex - target_ex, pos_ex + LIMITS[EXTRACT], 25)
                if q > 0:
                    result[EXTRACT].append(Order(EXTRACT, bb, -q))
        elif pos_ex < target_ex - HEDGE_BAND:
            bb, ba = _best(d_ex)
            if ba is not None:
                room_buy = LIMITS[EXTRACT] - pos_ex
                q = min(target_ex - pos_ex, room_buy, 25)
                if q > 0:
                    result[EXTRACT].append(Order(EXTRACT, ba, q))

        # ATM option: lean long gamma when gamma elevated — quote bid inside spread
        if gamma >= GAMMA_THRESH and d_opt:
            bb, ba = _best(d_opt)
            if bb is not None and ba is not None:
                pos = int(state.position.get(sym, 0))
                room_buy = LIMITS[sym] - pos
                room_sell = LIMITS[sym] + pos
                tick = state.timestamp // REQUOTE_EVERY
                if tick != int(store.get("opt_tick", -1)):
                    store["opt_tick"] = tick
                    bid_px = min(bb + MM_EDGE, ba - 1)
                    sell_px = max(ba - MM_EDGE, bb + 1)
                    if room_buy > 0:
                        result[sym].append(Order(sym, int(bid_px), min(OPTION_SIZE, room_buy)))
                    if room_sell > 0 and pos > 0:
                        result[sym].append(Order(sym, int(sell_px), -min(OPTION_SIZE, pos, room_sell)))

        # Hydrogel: wide passive MM (allowed product; low footprint)
        d_h = state.order_depths.get(HYDRO)
        m_h = _mid(d_h)
        if m_h is not None and d_h:
            bb, ba = _best(d_h)
            if bb is not None and ba is not None and ba - bb >= 2 * HYDRO_EDGE:
                ph = int(state.position.get(HYDRO, 0))
                if LIMITS[HYDRO] - ph > 0:
                    result[HYDRO].append(Order(HYDRO, int(m_h - HYDRO_EDGE), min(HYDRO_SIZE, LIMITS[HYDRO] - ph)))
                if ph > -LIMITS[HYDRO]:
                    q_s = min(HYDRO_SIZE, ph + LIMITS[HYDRO])
                    if q_s > 0:
                        result[HYDRO].append(Order(HYDRO, int(m_h + HYDRO_EDGE), -q_s))

        return result, conversions, json.dumps(store)
