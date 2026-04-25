"""
r3v_volume_weighted_residual_05 v19 — Parent: v8.

Same concept: fade IV minus volume-WLS quadratic smile residual.

Data-driven updates (from residual_predictive_idio_move.json):
- Per-symbol residual tails differ strongly; use symbol-specific entry thresholds.
- Predictive sign is asymmetric by strike on this strip:
  * 5200/5300/5500: positive residual more reliably mean-reverts next tick
    (idiosyncratic move tends negative) → prefer short entries.
  * 5100/5400: negative residual has clearer positive-revert signal → prefer longs.
- Keep all v8 guardrails intact (EMA smoothing, volume gate, spread gate, hydrogel).
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
TRADE_VOU = [f"VEV_{k}" for k in (5100, 5200, 5300, 5400, 5500)]

LIMITS = {H: 200, U: 200, **{v: 300 for v in VOU}}

_ANCHOR_S0 = {5250.0: 0, 5245.0: 1, 5267.5: 2}

RESID_ENTER = 0.0145
RESID_EXIT = 0.0045
EMA_ALPHA = 0.12
MIN_TOP_SUM_VOL = 28

MAX_Q = 12
MAX_SPREAD = 8
HYDRO_MIN_SPREAD = 14
HYDRO_EDGE = 4
HYDRO_Q = 5

_EMA_RES_KEY = "res_ema_by_vev"

# From residual_predictive_idio_move.json quantiles (approx 0.93 abs-resid quantile per symbol)
_RESID_ENTER_BY_SYM = {
    "VEV_5100": 0.0037,
    "VEV_5200": 0.0057,
    "VEV_5300": 0.0091,
    "VEV_5400": 0.0155,
    "VEV_5500": 0.0065,
}

# Directional asymmetry from tape predictive analysis.
_ALLOW_SHORT = {"VEV_5100": False, "VEV_5200": True, "VEV_5300": True, "VEV_5400": False, "VEV_5500": True}
_ALLOW_LONG = {"VEV_5100": True, "VEV_5200": False, "VEV_5300": False, "VEV_5400": True, "VEV_5500": False}


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

    for sym in TRADE_VOU:
        d = depths.get(sym)
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


def _update_res_ema(store: dict[str, Any], sym: str, raw_resid: float) -> float:
    ema_map = store.get(_EMA_RES_KEY)
    if not isinstance(ema_map, dict):
        ema_map = {}
    prev = ema_map.get(sym)
    if prev is not None and isinstance(prev, (int, float)) and math.isfinite(float(prev)):
        e = EMA_ALPHA * raw_resid + (1.0 - EMA_ALPHA) * float(prev)
    else:
        e = raw_resid
    ema_map[str(sym)] = e
    store[_EMA_RES_KEY] = ema_map
    return e


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
        orders: dict[str, list[Order]] = {p: [] for p in [H] + VOU}

        for sym in TRADE_VOU:
            lim = LIMITS[sym]
            pos = state.position.get(sym, 0)
            d = state.order_depths.get(sym)
            if d is None or sym not in residuals:
                continue
            iv, resid, _w = residuals[sym]
            res_ema = _update_res_ema(store, sym, resid)
            bb, ba = _best_bid_ask(d)
            if bb is None or ba is None or ba - bb > MAX_SPREAD:
                continue
            K = float(sym.split("_")[1])
            veg = vega(S, K, T, iv)
            q = min(MAX_Q, max(1, int(20.0 / max(veg, 1e-6))))
            q = min(q, lim - pos, lim + pos)
            if q <= 0:
                continue

            bv, av = _top_book_size(d)
            top_sum = bv + av

            # Flatten on raw residual (faster). New risk on EMA + liquidity.
            enter_thr = _RESID_ENTER_BY_SYM.get(sym, RESID_ENTER)
            allow_short = _ALLOW_SHORT.get(sym, True)
            allow_long = _ALLOW_LONG.get(sym, True)

            if abs(resid) < RESID_EXIT and abs(pos) > 0:
                if pos > 0:
                    orders[sym].append(Order(sym, ba, -min(q, pos)))
                elif pos < 0:
                    orders[sym].append(Order(sym, bb, min(q, -pos)))
            elif allow_short and res_ema > enter_thr and pos > -lim + q and top_sum >= MIN_TOP_SUM_VOL:
                orders[sym].append(Order(sym, bb, -q))
            elif allow_long and res_ema < -enter_thr and pos < lim - q and top_sum >= MIN_TOP_SUM_VOL:
                orders[sym].append(Order(sym, ba, q))

        dh = state.order_depths.get(H)
        if dh is not None:
            pos_h = state.position.get(H, 0)
            bbh, bah = _best_bid_ask(dh)
            if bbh is not None and bah is not None and bah - bbh >= HYDRO_MIN_SPREAD:
                if pos_h < LIMITS[H] - HYDRO_Q:
                    orders[H].append(Order(H, bbh + HYDRO_EDGE, HYDRO_Q))
                if pos_h > -LIMITS[H] + HYDRO_Q:
                    orders[H].append(Order(H, bah - HYDRO_EDGE, -HYDRO_Q))

        out = {k: v for k, v in orders.items() if v}
        return out, 0, json.dumps(store, separators=(",", ":"))
