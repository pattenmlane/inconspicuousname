"""
Round 3 — joint tight gate + vega rail (iteration 31).

Parent: v25. Shared thesis: when VEV_5200 and VEV_5300 BBO spreads are both <= 2 (same
timestamp), use as risk-on regime: scale VEV vega-rail room up; scale extract
two-sided size up only in that regime. Taper when either book is wide. No
hydrogel quotes (PnL focus: extract + VEVs only). See
round3work/vouchers_final_strategy/STRATEGY.txt; tape gate frequency:
analysis_outputs/r3_tight_5200_5300_gate_by_day.json.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy.stats import norm

from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"

COEFFS_HIGH_TO_LOW = [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055]

# --- tunables (iteration 31) ---
VEGA_RAIL = 650.0
# Joint book regime (BBO: ask₁ − bid₁)
TIGHT_MAX_SPREAD = 2
VEV_GATE_A = "VEV_5200"
VEV_GATE_B = "VEV_5300"
# Continuous sizing (no hard skip): tight gate boosts VEV vega room; wide tapers
TIGHT_VEGA_ROOM_MULT = 1.3
WIDE_VEGA_ROOM_MULT = 0.88
# P_nL focus: no extract MM in this version (STRATEGY.txt: mid edge != bid/ask PnL)
TIGHT_EXTRACT_MM_Q = 0
K_OVER_S_MIN = 0.97
K_OVER_S_MAX = 1.03
THR_OPEN = 0.35
# Per-strike beta (same-step) from analysis_outputs/extract_move_propagation_by_contract.json
BETA_BY_STRIKE: dict[int, float] = {
    4000: 0.7451, 4500: 0.6618, 5000: 0.6535, 5100: 0.5773, 5200: 0.4367,
    5300: 0.2727, 5400: 0.1289, 5500: 0.0549, 6000: 0.0, 6500: 0.0,
}
BETA_REF = 0.42
THR_SCALE_MIN = 0.82
THR_SCALE_MAX = 1.2
THR_CLOSE = 0.0
LOW_VEGA_THR_ADJ = 0.5
IV_SCALPING_THR = 0.45
QUIET_FLATTEN = False
LOW_VEGA_CUTOFF = 1.0
THEO_NORM_WINDOW = 20
IV_SCALPING_WINDOW = 100
WARMUP_STEPS = 10
DELTA_HEDGE_FRAC = 0.0
EXTRACT_EDGE = 2
HYDRO_EDGE = 2
EXTRACT_MM_Q = 0
HYDRO_MM_Q = 0
HYDRO_QUOTE = 0


def _cdf(x: float) -> float:
    return float(norm.cdf(x))


def _pdf(x: float) -> float:
    return float(norm.pdf(x))


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def dte_effective(day: int, timestamp: int) -> float:
    prog = (int(timestamp) // 100) / 10_000.0
    return max(float(dte_from_csv_day(day)) - prog, 1e-6)


def t_years_effective(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def iv_smile(S: float, K: float, T: float) -> float:
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    m = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(np.asarray(COEFFS_HIGH_TO_LOW, dtype=float), m))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> tuple[float, float]:
    if T <= 0 or sigma <= 1e-12:
        return max(S - K, 0.0), 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    return float(price), float(_cdf(d1))


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(S * _pdf(d1) * math.sqrt(T))


def book_from_depth(
    depth: OrderDepth,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """(bid_wall, ask_wall, best_bid, best_ask, wall_mid) — Frankfurt wall semantics."""
    buys = depth.buy_orders or {}
    sells = depth.sell_orders or {}
    if not buys or not sells:
        return None, None, None, None, None
    bid_prices = [int(p) for p in buys]
    ask_prices = [int(p) for p in sells]
    bid_wall = float(min(bid_prices))
    ask_wall = float(max(ask_prices))
    best_bid = float(max(bid_prices))
    best_ask = float(min(ask_prices))
    wall_mid = 0.5 * (bid_wall + ask_wall)
    return bid_wall, ask_wall, best_bid, best_ask, wall_mid


def synthetic_walls(
    bid_wall: float | None,
    ask_wall: float | None,
    best_bid: float | None,
    best_ask: float | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Returns (best_bid, best_ask, wall_mid) after filling missing walls; also returns walls."""
    if bid_wall is None and ask_wall is not None:
        bw = float(ask_wall) - 1.0
        aw = float(ask_wall)
        wm = aw - 0.5
        bb = bw
        ba = aw
        return bw, aw, bb, ba, wm
    if ask_wall is None and bid_wall is not None:
        bw = float(bid_wall)
        aw = float(bid_wall) + 1.0
        wm = bw + 0.5
        bb = bw
        ba = aw
        return bw, aw, bb, ba, wm
    if bid_wall is not None and ask_wall is not None:
        wm = 0.5 * (float(bid_wall) + float(ask_wall))
        return float(bid_wall), float(ask_wall), best_bid, best_ask, wm
    return bid_wall, ask_wall, best_bid, best_ask, None


def bbo_spread_int(depth: OrderDepth | None) -> int | None:
    if depth is None:
        return None
    bu = depth.buy_orders or {}
    se = depth.sell_orders or {}
    if not bu or not se:
        return None
    bidp = int(max(bu))
    askp = int(min(se))
    if askp < bidp:
        return None
    return int(askp - bidp)


def joint_tight_gate(depths: dict[str, OrderDepth], sym52: str | None, sym53: str | None) -> bool:
    if sym52 is None or sym53 is None:
        return False
    s1 = bbo_spread_int(depths.get(sym52))
    s2 = bbo_spread_int(depths.get(sym53))
    if s1 is None or s2 is None:
        return False
    return s1 <= TIGHT_MAX_SPREAD and s2 <= TIGHT_MAX_SPREAD


def ema(store: dict[str, float], key: str, window: int, value: float) -> float:
    old = float(store.get(key, 0.0))
    alpha = 2.0 / (window + 1.0)
    new = alpha * value + (1.0 - alpha) * old
    store[key] = new
    return new


def effective_thr_open(K: int) -> float:
    b = BETA_BY_STRIKE.get(K, 0.35)
    b = max(b, 0.08)
    scale = BETA_REF / b
    scale = max(THR_SCALE_MIN, min(THR_SCALE_MAX, scale))
    return max(0.05, THR_OPEN * scale)


def iv_scalp_orders(
    wall_mid: float,
    best_bid: float,
    best_ask: float,
    mean_theo_diff: float,
    current_theo_diff: float,
    switch_mean: float,
    vega: float,
    pos: int,
    max_buy: int,
    max_sell: int,
    thr_open_use: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    bids: list[tuple[int, int]] = []
    asks: list[tuple[int, int]] = []
    low_vega_adj = LOW_VEGA_THR_ADJ if vega <= LOW_VEGA_CUTOFF else 0.0
    if switch_mean >= IV_SCALPING_THR:
        if current_theo_diff - wall_mid + best_bid - mean_theo_diff >= (thr_open_use + low_vega_adj) and max_sell > 0:
            asks.append((int(best_bid), max_sell))
        if current_theo_diff - wall_mid + best_bid - mean_theo_diff >= THR_CLOSE and pos > 0:
            asks.append((int(best_bid), pos))
        elif current_theo_diff - wall_mid + best_ask - mean_theo_diff <= -(thr_open_use + low_vega_adj) and max_buy > 0:
            bids.append((int(best_ask), max_buy))
        if current_theo_diff - wall_mid + best_ask - mean_theo_diff <= -THR_CLOSE and pos < 0:
            bids.append((int(best_ask), -pos))
    else:
        if QUIET_FLATTEN:
            if pos > 0:
                asks.append((int(best_bid), pos))
            elif pos < 0:
                bids.append((int(best_ask), -pos))
    return bids, asks


def sym_for(state: TradingState, product: str) -> str | None:
    for s, lst in (state.listings or {}).items():
        if getattr(lst, "product", None) == product:
            return s
    return None


def merge_orders(orders: dict[str, list[Order]]) -> dict[str, list[Order]]:
    out: dict[str, list[Order]] = {}
    for sym, lst in orders.items():
        acc: dict[int, int] = {}
        for o in lst:
            acc[o.price] = acc.get(o.price, 0) + o.quantity
        merged = [Order(sym, p, q) for p, q in acc.items() if q != 0]
        if merged:
            out[sym] = merged
    return out


class Trader:
    def run(self, state: TradingState):
        td_raw = getattr(state, "traderData", "") or ""
        try:
            bag: dict[str, Any] = json.loads(td_raw) if td_raw else {}
        except json.JSONDecodeError:
            bag = {}
        if not isinstance(bag, dict):
            bag = {}

        last_ts = int(bag.get("_last_ts", -1))
        ts = int(getattr(state, "timestamp", 0))
        csv_day = int(bag.get("_csv_day", 0))
        if last_ts >= 0 and ts < last_ts:
            csv_day += 1
        bag["_last_ts"] = ts
        bag["_csv_day"] = csv_day

        ema_store: dict[str, float] = bag.get("ema") if isinstance(bag.get("ema"), dict) else {}
        ema_store = {str(k): float(v) for k, v in ema_store.items() if isinstance(v, (int, float))}

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}

        sym_u = sym_for(state, UNDERLYING)
        sym_h = sym_for(state, HYDRO)
        sym_52 = sym_for(state, VEV_GATE_A)
        sym_53 = sym_for(state, VEV_GATE_B)
        tight = joint_tight_gate(depths, sym_52, sym_53)
        room_mult = TIGHT_VEGA_ROOM_MULT if tight else WIDE_VEGA_ROOM_MULT
        if sym_u is None:
            bag["ema"] = ema_store
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        depth_u = depths.get(sym_u)
        if depth_u is None:
            bag["ema"] = ema_store
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        _ubw, _uaw, ubb, uba, wmu = book_from_depth(depth_u)
        if wmu is None or ubb is None or uba is None:
            bag["ema"] = ema_store
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        S = 0.5 * (ubb + uba)
        T = t_years_effective(csv_day, ts)

        orders_out: dict[str, list[Order]] = {}
        lim_u = 200
        lim_h = 200

        # --- partial delta hedge (options -> extract) ---
        net_opt_delta = 0.0
        if ts // 100 >= WARMUP_STEPS:
            for v in VOUCHERS:
                sym_v = sym_for(state, v)
                if sym_v is None:
                    continue
                K = int(v.split("_")[1])
                if K / S < K_OVER_S_MIN or K / S > K_OVER_S_MAX:
                    continue
                dv = depths.get(sym_v)
                if dv is None:
                    continue
                bw, aw, bb, ba, _wm0 = book_from_depth(dv)
                bwf, awf, bb2, ba2, wmf = synthetic_walls(bw, aw, bb, ba)
                if wmf is None or bb2 is None or ba2 is None:
                    continue
                sig = iv_smile(S, K, T)
                if not math.isfinite(sig) or sig <= 0:
                    continue
                _, delt = bs_call(S, K, T, sig)
                net_opt_delta += delt * float(pos.get(sym_v, 0))

        hedge = int(round(DELTA_HEDGE_FRAC * net_opt_delta))
        hedge = max(-lim_u, min(lim_u, hedge))
        pos_u = int(pos.get(sym_u, 0))
        if hedge != 0 and ts // 100 >= WARMUP_STEPS:
            tgt = -hedge
            dq = tgt - pos_u
            if dq != 0:
                q = max(-lim_u - pos_u, min(lim_u - pos_u, dq))
                if q > 0 and uba is not None:
                    orders_out.setdefault(sym_u, []).append(Order(sym_u, int(uba), q))
                elif q < 0 and ubb is not None:
                    orders_out.setdefault(sym_u, []).append(Order(sym_u, int(ubb), q))

        # --- extract (joint-tight only) + optional hydrogel off for PnL focus ---
        if ts // 100 >= WARMUP_STEPS:
            mid_u = 0.5 * (ubb + uba)
            bu = int(mid_u) - EXTRACT_EDGE
            su = int(mid_u) + EXTRACT_EDGE
            qu = TIGHT_EXTRACT_MM_Q if tight else 0
            if qu > 0:
                if pos_u + qu <= lim_u:
                    orders_out.setdefault(sym_u, []).append(Order(sym_u, bu, qu))
                if pos_u - qu >= -lim_u:
                    orders_out.setdefault(sym_u, []).append(Order(sym_u, su, -qu))

            if sym_h is not None and HYDRO_QUOTE > 0:
                dh = depths.get(sym_h)
                if dh is not None:
                    _hbw, _haw, hb, ha, wmh = book_from_depth(dh)
                    if wmh is not None and hb is not None and ha is not None:
                        mid_h = 0.5 * (hb + ha)
                        pos_h = int(pos.get(sym_h, 0))
                        bh = int(mid_h) - HYDRO_EDGE
                        sh = int(mid_h) + HYDRO_EDGE
                        qh = HYDRO_QUOTE
                        if pos_h + qh <= lim_h:
                            orders_out.setdefault(sym_h, []).append(Order(sym_h, bh, qh))
                        if pos_h - qh >= -lim_h:
                            orders_out.setdefault(sym_h, []).append(Order(sym_h, sh, -qh))

        if ts // 100 < WARMUP_STEPS:
            bag["ema"] = ema_store
            return merge_orders(orders_out), 0, json.dumps(bag, separators=(",", ":"))

        lim_v = 300

        for v in VOUCHERS:
            sym_v = sym_for(state, v)
            if sym_v is None:
                continue
            K = int(v.split("_")[1])
            if K / S < K_OVER_S_MIN or K / S > K_OVER_S_MAX:
                continue
            dv = depths.get(sym_v)
            if dv is None:
                continue
            bw, aw, bb, ba, _wm0 = book_from_depth(dv)
            bwf, awf, bb2, ba2, wmf = synthetic_walls(bw, aw, bb, ba)
            if wmf is None or bb2 is None or ba2 is None:
                continue
            wall_mid = float(wmf)
            best_bid = float(bb2)
            best_ask = float(ba2)
            sig = iv_smile(S, K, T)
            if not math.isfinite(sig) or sig <= 0:
                continue
            theo, delta = bs_call(S, K, T, sig)
            vega = bs_vega(S, K, T, sig)
            option_theo_diff = wall_mid - theo
            ema(ema_store, f"{v}_theo_diff", THEO_NORM_WINDOW, option_theo_diff)
            mean_diff = float(ema_store.get(f"{v}_theo_diff", 0.0))
            ema(ema_store, f"{v}_avg_devs", IV_SCALPING_WINDOW, abs(option_theo_diff - mean_diff))
            switch_mean = float(ema_store.get(f"{v}_avg_devs", 0.0))

            pos_v = int(pos.get(sym_v, 0))
            abs_p = abs(pos_v)
            vega_cap = int(VEGA_RAIL / max(vega, 0.05))
            cap_adj = int(max(1, round(vega_cap * room_mult)))
            room = max(0, min(lim_v, cap_adj) - abs_p)
            max_buy = min(lim_v - pos_v, room)
            max_sell = min(lim_v + pos_v, room)

            thr_use = effective_thr_open(K)
            bids, asks = iv_scalp_orders(
                wall_mid,
                best_bid,
                best_ask,
                mean_diff,
                option_theo_diff,
                switch_mean,
                vega,
                pos_v,
                max_buy,
                max_sell,
                thr_use,
            )
            ol: list[Order] = []
            for px, q in bids:
                q = min(q, lim_v - pos_v)
                if q > 0:
                    ol.append(Order(sym_v, px, q))
            for px, q in asks:
                q = min(q, lim_v + pos_v)
                if q > 0:
                    ol.append(Order(sym_v, px, -q))
            if ol:
                orders_out.setdefault(sym_v, []).extend(ol)

        bag["ema"] = ema_store
        return merge_orders(orders_out), 0, json.dumps(bag, separators=(",", ":"))
