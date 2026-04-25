"""
r3v_volume_weighted_residual_05 v1 — Same IV-residual fade as v0, parameter sweep:

- No HYDROGEL_PACK (v0 hydrogel PnL dominated; refocus on voucher thesis).
- Higher RESID_ENTER / wider RESID_EXIT (fewer, cleaner fades).
- Top-of-book spread filter (skip illiquid strikes).
- Light delta hedge on VELVETFRUIT_EXTRACT toward portfolio Black–Scholes delta.

Timing / anchors: same as trader_v0.py.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]

LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in VOU},
}

_ANCHOR_S0 = {5250.0: 0, 5245.0: 1, 5267.5: 2}

RESID_ENTER = 0.02
RESID_EXIT = 0.007
MAX_SPREAD = 10
MAX_Q = 14
VEGA_NOM = 22.0
HEDGE_BAND = 25


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def dte_open(csv_day: int) -> float:
    return float(8 - csv_day)


def dte_eff(csv_day: int, ts: int) -> float:
    return max(dte_open(csv_day) - (int(ts) // 100) / 10_000.0, 1e-6)


def t_years(csv_day: int, ts: int) -> float:
    return dte_eff(csv_day, ts) / 365.0


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sig <= 1e-12:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


def implied_vol(mid: float, S: float, K: float, T: float, r: float = 0.0) -> float | None:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-9 or mid >= S - 1e-9 or S <= 0 or K <= 0 or T <= 0:
        return None
    lo, hi = 1e-4, 10.0
    for _ in range(64):
        m = 0.5 * (lo + hi)
        if bs_call(S, K, T, m, r) > mid:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def call_delta(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0 or sig <= 1e-12:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    return float(_ncdf(d1))


def vega(S: float, K: float, T: float, sig: float, r: float = 0.0) -> float:
    if T <= 0 or sig <= 1e-12:
        return 0.0
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    return float(S * _npdf(d1) * math.sqrt(T))


def _best_bid_ask(d: OrderDepth) -> tuple[int | None, int | None]:
    if not d.buy_orders or not d.sell_orders:
        return None, None
    return max(d.buy_orders), min(d.sell_orders)


def _top_book_size(d: OrderDepth) -> tuple[int, int]:
    bb, ba = _best_bid_ask(d)
    if bb is None or ba is None:
        return 0, 0
    bv = int(d.buy_orders.get(bb, 0))
    av = int(abs(d.sell_orders.get(ba, 0)))
    return bv, av


def infer_csv_day(state: TradingState, store: dict[str, Any]) -> int:
    if "csv_day" in store:
        return int(store["csv_day"])
    du = state.order_depths.get(U)
    if du is None:
        return 0
    bb, ba = _best_bid_ask(du)
    if bb is None or ba is None:
        return 0
    s0 = 0.5 * (bb + ba)
    best_d, best_e = 0, 1e9
    for s_anchor, d in _ANCHOR_S0.items():
        e = abs(s0 - s_anchor)
        if e < best_e:
            best_e, best_d = e, d
    if best_e > 5.0:
        best_d = 0
    store["csv_day"] = best_d
    return best_d


def fit_smile_residuals(
    S: float, T: float, depths: dict[str, OrderDepth]
) -> dict[str, tuple[float, float, float]]:
    if S <= 0 or T <= 0:
        return {}
    sqrtT = math.sqrt(T)
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    meta: list[tuple[str, float, float, float]] = []

    for sym in VOU:
        d = depths.get(sym)
        if d is None:
            continue
        bb, ba = _best_bid_ask(d)
        if bb is None or ba is None:
            continue
        if ba - bb > MAX_SPREAD:
            continue
        mid = 0.5 * (bb + ba)
        K = float(sym.split("_")[1])
        iv = implied_vol(mid, S, K, T)
        if iv is None:
            continue
        bv, av = _top_book_size(d)
        w = float(max(bv + av + 1, 1))
        m_t = math.log(K / S) / sqrtT
        xs.append(m_t)
        ys.append(iv)
        ws.append(w)
        meta.append((sym, iv, m_t, w))

    if len(xs) < 5:
        return {}

    X = np.c_[np.array(xs) ** 2, xs, np.ones(len(xs))]
    wsqrt = np.sqrt(np.array(ws))
    coef, *_ = np.linalg.lstsq(X * wsqrt[:, None], np.array(ys) * wsqrt, rcond=None)
    out: dict[str, tuple[float, float, float]] = {}
    for sym, iv, m_t, w in meta:
        pred = float(coef[0] * m_t * m_t + coef[1] * m_t + coef[2])
        out[sym] = (iv, iv - pred, w)
    return out


class Trader:
    def run(self, state: TradingState):
        store = _parse_td(state.traderData)
        csv_day = infer_csv_day(state, store)
        T = t_years(csv_day, state.timestamp)

        du = state.order_depths.get(U)
        if du is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        bb_u, ba_u = _best_bid_ask(du)
        if bb_u is None or ba_u is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        S = 0.5 * (bb_u + ba_u)

        residuals = fit_smile_residuals(S, T, state.order_depths)
        orders: dict[str, list[Order]] = {p: [] for p in [U] + VOU}

        port_delta = 0.0
        for sym in VOU:
            pos = state.position.get(sym, 0)
            if pos == 0:
                continue
            d = state.order_depths.get(sym)
            if d is None:
                continue
            bb, ba = _best_bid_ask(d)
            if bb is None or ba is None or ba - bb > MAX_SPREAD:
                continue
            mid = 0.5 * (bb + ba)
            K = float(sym.split("_")[1])
            iv = implied_vol(mid, S, K, T)
            if iv is None:
                continue
            port_delta += pos * call_delta(S, K, T, iv)

        pos_u = state.position.get(U, 0)
        target_u = int(round(-port_delta))
        if target_u > LIMITS[U]:
            target_u = LIMITS[U]
        if target_u < -LIMITS[U]:
            target_u = -LIMITS[U]

        if pos_u < target_u - HEDGE_BAND and pos_u < LIMITS[U] - 5:
            q = min(30, target_u - pos_u, LIMITS[U] - pos_u)
            if q > 0:
                orders[U].append(Order(U, ba_u, q))
        elif pos_u > target_u + HEDGE_BAND and pos_u > -LIMITS[U] + 5:
            q = min(30, pos_u - target_u, pos_u + LIMITS[U])
            if q > 0:
                orders[U].append(Order(U, bb_u, -q))

        for sym in VOU:
            lim = LIMITS[sym]
            pos = state.position.get(sym, 0)
            d = state.order_depths.get(sym)
            if d is None or sym not in residuals:
                continue
            iv, resid, _w = residuals[sym]
            bb, ba = _best_bid_ask(d)
            if bb is None or ba is None or ba - bb > MAX_SPREAD:
                continue
            K = float(sym.split("_")[1])
            veg = vega(S, K, T, iv)
            q = min(MAX_Q, max(1, int(VEGA_NOM / max(veg, 1e-6))))
            q = min(q, lim - pos, lim + pos)

            if resid > RESID_ENTER and pos > -lim + q:
                orders[sym].append(Order(sym, bb, -q))
            elif resid < -RESID_ENTER and pos < lim - q:
                orders[sym].append(Order(sym, ba, q))
            elif abs(resid) < RESID_EXIT and abs(pos) > 0:
                if pos > 0:
                    orders[sym].append(Order(sym, ba, -min(q, pos)))
                elif pos < 0:
                    orders[sym].append(Order(sym, bb, min(q, -pos)))

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
