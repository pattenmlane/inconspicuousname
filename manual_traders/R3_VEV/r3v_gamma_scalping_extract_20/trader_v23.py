"""
v23: v20 base + **soft** joint 5200/5300 spread gate (STRATEGY.txt TH=2).

Unlike v22 (hard filter: no VEV quotes unless both legs ≤2), we **always** quote
when gamma passes, but **scale option clip** by regime: **1.0×** when the joint
gate is tight, **0.5×** when either leg is wide (floor 1). **Caveat:** 0.5× on
5 and 6 base clips rounds to 3 and 3 — same as unscaled 5/6 for half the grid,
so in backtests v23 can match v22/v20 unless you use a floor multiplier that
splits 5/6 (see v24). TTE: tape days 0..3.
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
GATE_S5200 = "VEV_5200"
GATE_S5300 = "VEV_5300"
SPREAD_TH = 2
CLIP_MULT_TIGHT = 1.0
CLIP_MULT_WIDE = 0.5

PRODUCTS = [HYDRO, EXTRACT] + VEV

LIMITS = {
    HYDRO: 200,
    EXTRACT: 200,
    **{v: 300 for v in VEV},
}

TTE_BY_TAPE_DAY = (8, 7, 6, 5)
DAYS_PER_YEAR = 365.0
R_RATE = 0.0

GAMMA_THRESH = 8e-5
GAMMA_THRESH_DAY1 = 6.5e-5
OPTION_SIZE = 10
REQUOTE_EVERY = 3
MM_EDGE = 1
DELTA_HEDGE_FRAC = 0.78
HEDGE_BAND = 6
DAY1_HEDGE_FRAC = 0.48
DAY1_HEDGE_BAND = 12
DAY1_CLIP_HIGH_DELTA = 4
DAY1_CLIP_LOW_DELTA = 6


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


def _bbo_spread(depth: OrderDepth | None) -> int | None:
    bb, ba = _best(depth)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def _joint_tight_gate(state: TradingState) -> bool:
    s52 = _bbo_spread(state.order_depths.get(GATE_S5200))
    s53 = _bbo_spread(state.order_depths.get(GATE_S5300))
    if s52 is None or s53 is None:
        return False
    return s52 <= SPREAD_TH and s53 <= SPREAD_TH


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


def _scale_clip(base: int, mult: float) -> int:
    return max(1, int(round(float(base) * mult)))


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

        td = min(tape_day, len(TTE_BY_TAPE_DAY) - 1)
        tte_days = TTE_BY_TAPE_DAY[td]
        T = tte_days / DAYS_PER_YEAR

        d_ex = state.order_depths.get(EXTRACT)
        S = _wall_mid(d_ex) or _mid(d_ex)
        if S is None:
            return result, conversions, json.dumps(store)

        gate_tight = _joint_tight_gate(state)
        store["tight_5200_5300"] = gate_tight
        clip_mult = CLIP_MULT_TIGHT if gate_tight else CLIP_MULT_WIDE

        k1, k2 = two_nearest_strikes(S)
        syms = (f"VEV_{k1}", f"VEV_{k2}")

        portfolio_delta = 0.0
        max_gamma = 0.0
        iv_map: dict[str, float] = {}
        delta_map: dict[str, float] = {}

        for sym in syms:
            K = float(sym.split("_")[1])
            d_opt = state.order_depths.get(sym)
            mid_c = _mid(d_opt)
            if mid_c is None:
                continue
            iv = implied_vol(mid_c, S, K, T)
            if iv is None:
                continue
            key = f"iv_ema_{sym}"
            ema = store.get(key)
            if ema is None:
                ema = iv
            else:
                ema = 0.12 * iv + 0.88 * float(ema)
            store[key] = ema
            iv_map[sym] = float(ema)
            dlt, gam = bs_delta_gamma(S, K, T, float(ema))
            delta_map[sym] = float(dlt)
            pos = int(state.position.get(sym, 0))
            portfolio_delta += pos * dlt
            max_gamma = max(max_gamma, gam)

        store["last_gamma"] = max_gamma
        store["port_delta"] = portfolio_delta

        pos_ex = int(state.position.get(EXTRACT, 0))
        if tape_day < 2:
            hedge_frac = DAY1_HEDGE_FRAC if tape_day == 1 else DELTA_HEDGE_FRAC
            hedge_band = DAY1_HEDGE_BAND if tape_day == 1 else HEDGE_BAND
            target_ex = int(round(-hedge_frac * portfolio_delta))
            target_ex = max(-LIMITS[EXTRACT], min(LIMITS[EXTRACT], target_ex))
            if pos_ex > target_ex + hedge_band:
                bb, ba = _best(d_ex)
                if bb is not None:
                    q = min(pos_ex - target_ex, pos_ex + LIMITS[EXTRACT], 18)
                    if q > 0:
                        result[EXTRACT].append(Order(EXTRACT, bb, -q))
            elif pos_ex < target_ex - hedge_band:
                bb, ba = _best(d_ex)
                if ba is not None:
                    room_buy = LIMITS[EXTRACT] - pos_ex
                    q = min(target_ex - pos_ex, room_buy, 18)
                    if q > 0:
                        result[EXTRACT].append(Order(EXTRACT, ba, q))

        gthr = GAMMA_THRESH_DAY1 if tape_day == 1 else GAMMA_THRESH
        if max_gamma < gthr:
            return result, conversions, json.dumps(store)

        tick = state.timestamp // REQUOTE_EVERY
        if tick == int(store.get("opt_tick", -1)):
            return result, conversions, json.dumps(store)
        store["opt_tick"] = tick

        per = max(1, OPTION_SIZE // 2)
        per_by_sym = {sym: per for sym in syms}
        if tape_day == 1 and len(syms) == 2:
            s0, s1 = syms
            d0 = delta_map.get(s0)
            d1 = delta_map.get(s1)
            if d0 is not None and d1 is not None:
                if d0 > d1:
                    per_by_sym[s0] = DAY1_CLIP_LOW_DELTA
                    per_by_sym[s1] = DAY1_CLIP_HIGH_DELTA
                elif d1 > d0:
                    per_by_sym[s1] = DAY1_CLIP_LOW_DELTA
                    per_by_sym[s0] = DAY1_CLIP_HIGH_DELTA
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
            base_clip = per_by_sym.get(sym, per)
            clip = _scale_clip(base_clip, clip_mult)
            if room_buy > 0:
                result[sym].append(Order(sym, int(bid_px), min(clip, room_buy)))
            if room_sell > 0 and pos > 0:
                result[sym].append(Order(sym, int(sell_px), -min(clip, pos, room_sell)))

        return result, conversions, json.dumps(store)
