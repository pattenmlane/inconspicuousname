"""
v11: v4 with per-strike clip skew by instantaneous IV rank between the two
nearest strikes (higher IV -> clip 6, lower -> 4; else equal 5), informed by
two-strike IV skew on tapes (higher K often carries a bit more IV in days 1–2).
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

TTE_BY_TAPE_DAY = (8, 7, 6)
DAYS_PER_YEAR = 365.0
R_RATE = 0.0

GAMMA_THRESH = 8e-5
OPTION_SIZE = 10
OPTION_CLIP_HI = 6
OPTION_CLIP_LO = 4
REQUOTE_EVERY = 3
MM_EDGE = 1
DELTA_HEDGE_FRAC = 0.78
HEDGE_BAND = 6


def _wall_mid(depth: OrderDepth | None) -> float | None:
    if not depth or not depth.buy_orders or not depth.sell_orders:
        return None
    bid_wall = min(depth.buy_orders.keys())
    ask_wall = max(depth.sell_orders.keys())
    return (bid_wall + ask_wall) / 2.0


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


def two_nearest_strikes(S: float) -> tuple[int, int]:
    ranked = sorted(STRIKES, key=lambda x: abs(float(x) - S))
    a, b = ranked[0], ranked[1]
    return (a, b) if a < b else (b, a)


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
        S = _wall_mid(d_ex) or _mid(d_ex)
        if S is None:
            return result, conversions, json.dumps(store)

        k1, k2 = two_nearest_strikes(S)
        syms = (f"VEV_{k1}", f"VEV_{k2}")

        portfolio_delta = 0.0
        max_gamma = 0.0
        iv_map: dict[str, float] = {}
        iv_inst: dict[str, float] = {}

        for sym in syms:
            K = float(sym.split("_")[1])
            d_opt = state.order_depths.get(sym)
            mid_c = _mid(d_opt)
            if mid_c is None:
                continue
            iv = implied_vol(mid_c, S, K, T)
            if iv is None:
                continue
            iv_inst[sym] = float(iv)
            key = f"iv_ema_{sym}"
            ema = store.get(key)
            if ema is None:
                ema = iv
            else:
                ema = 0.12 * iv + 0.88 * float(ema)
            store[key] = ema
            iv_map[sym] = float(ema)
            dlt, gam = bs_delta_gamma(S, K, T, float(ema))
            pos = int(state.position.get(sym, 0))
            portfolio_delta += pos * dlt
            max_gamma = max(max_gamma, gam)

        store["last_gamma"] = max_gamma
        store["port_delta"] = portfolio_delta

        pos_ex = int(state.position.get(EXTRACT, 0))
        if tape_day < 2:
            target_ex = int(round(-DELTA_HEDGE_FRAC * portfolio_delta))
            target_ex = max(-LIMITS[EXTRACT], min(LIMITS[EXTRACT], target_ex))
            if pos_ex > target_ex + HEDGE_BAND:
                bb, ba = _best(d_ex)
                if bb is not None:
                    q = min(pos_ex - target_ex, pos_ex + LIMITS[EXTRACT], 18)
                    if q > 0:
                        result[EXTRACT].append(Order(EXTRACT, bb, -q))
            elif pos_ex < target_ex - HEDGE_BAND:
                bb, ba = _best(d_ex)
                if ba is not None:
                    room_buy = LIMITS[EXTRACT] - pos_ex
                    q = min(target_ex - pos_ex, room_buy, 18)
                    if q > 0:
                        result[EXTRACT].append(Order(EXTRACT, ba, q))

        if max_gamma < GAMMA_THRESH:
            return result, conversions, json.dumps(store)

        tick = state.timestamp // REQUOTE_EVERY
        if tick == int(store.get("opt_tick", -1)):
            return result, conversions, json.dumps(store)
        store["opt_tick"] = tick

        s_lo, s_hi = syms[0], syms[1]
        v_lo, v_hi = iv_inst.get(s_lo), iv_inst.get(s_hi)
        if v_lo is not None and v_hi is not None:
            if v_lo > v_hi:
                per_by_sym = {s_lo: OPTION_CLIP_HI, s_hi: OPTION_CLIP_LO}
            elif v_hi > v_lo:
                per_by_sym = {s_lo: OPTION_CLIP_LO, s_hi: OPTION_CLIP_HI}
            else:
                per_by_sym = {s_lo: max(1, OPTION_SIZE // 2), s_hi: max(1, OPTION_SIZE // 2)}
        else:
            base = max(1, OPTION_SIZE // 2)
            per_by_sym = {s_lo: base, s_hi: base}
        for sym in syms:
            if sym not in iv_map:
                continue
            d_opt = state.order_depths.get(sym)
            if not d_opt:
                continue
            bb, ba = _best(d_opt)
            if bb is None or ba is None:
                continue
            pos = int(state.position.get(sym, 0))
            room_buy = LIMITS[sym] - pos
            room_sell = LIMITS[sym] + pos
            bid_px = min(bb + MM_EDGE, ba - 1)
            sell_px = max(ba - MM_EDGE, bb + 1)
            per = per_by_sym.get(sym, max(1, OPTION_SIZE // 2))
            if room_buy > 0:
                result[sym].append(Order(sym, int(bid_px), min(per, room_buy)))
            if room_sell > 0 and pos > 0:
                result[sym].append(Order(sym, int(sell_px), -min(per, pos, room_sell)))

        return result, conversions, json.dumps(store)
