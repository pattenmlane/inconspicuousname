"""
r3v_volume_weighted_residual_05 v22 — v21 + `VEV_5200`+`VEV_5300` joint spread gate (STRATEGY.txt, TH=2).

**Soft regime use (not a hard entry ban):** when both spreads are <= `JOINT_TH`, treat as
**clean surface / risk-on** and scale **new** voucher size with `GATE_TIGHT_Q_MULT`;
otherwise use `GATE_LOOSE_Q_MULT` (Sonic: wide 5200/5300 = different execution regime).
Exits/flatten on `RESID_EXIT` recompute full vega-`q` (no gate shrink) for urgency.

**Hydrogel:** same passive two-sided quotes as v8/v21 (not the optimization target; joint gate
applies to **VEV** residual fades only).
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"  # not quoted / not in PnL objective
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOU = [f"VEV_{k}" for k in STRIKES]
TRADE_VOU = [f"VEV_{k}" for k in (5100, 5200, 5300, 5400, 5500)]
SYM_5200 = "VEV_5200"
SYM_5300 = "VEV_5300"
SYM_5400 = "VEV_5400"

LIMITS = {H: 200, U: 200, **{v: 300 for v in VOU}}

_ANCHOR_S0 = {5250.0: 0, 5245.0: 1, 5267.5: 2}

# Joint book gate (same TH as `analyze_vev_5200_5300_tight_gate_r3.py` / STRATEGY.txt)
JOINT_TH = 2

RESID_ENTER = 0.0145
RESID_EXIT = 0.0045
EMA_ALPHA = 0.12
MIN_TOP_SUM_VOL = 28

SHOCK_ABS_DS = 2.0

# New-entry size: tight joint books -> multiply base q; else shrink (keeps some flow when wide).
GATE_TIGHT_Q_MULT = 1.0
GATE_LOOSE_Q_MULT = 0.5
GATE_LOOSE_Q_MIN = 1

MAX_Q = 12
MAX_SPREAD = 8
HYDRO_MIN_SPREAD = 14
HYDRO_EDGE = 4
HYDRO_Q = 5

_EMA_RES_KEY = "res_ema_by_vev"
_LAST_S_KEY = "last_U_mid_v22"


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


def _spread1(d: OrderDepth) -> int | None:
    bb, ba = _best_bid_ask(d)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def joint_tight_books(depths: dict[str, OrderDepth]) -> bool:
    d2 = depths.get(SYM_5200)
    d3 = depths.get(SYM_5300)
    if d2 is None or d3 is None:
        return False
    s2 = _spread1(d2)
    s3 = _spread1(d3)
    if s2 is None or s3 is None:
        return False
    return s2 <= JOINT_TH and s3 <= JOINT_TH


def _entry_sell_price(bb: int, ba: int, dS: float, stress: bool, sym: str) -> int:
    if stress and sym == SYM_5400 and ba - bb > 1:
        if dS > 0:
            return bb
        return min(bb + 1, ba - 1)
    return bb


def _entry_buy_price(bb: int, ba: int, dS: float, stress: bool, sym: str) -> int:
    if stress and sym == SYM_5400 and ba - bb > 1:
        return max(ba - 1, bb + 1)
    return ba


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

        prev = store.get(_LAST_S_KEY)
        if isinstance(prev, (int, float)) and math.isfinite(float(prev)):
            dS = S - float(prev)
            stress = abs(dS) >= SHOCK_ABS_DS
        else:
            dS = 0.0
            stress = False
        store[_LAST_S_KEY] = S

        gate = joint_tight_books(state.order_depths)

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
            q0 = min(MAX_Q, max(1, int(20.0 / max(veg, 1e-6))))
            mult = GATE_TIGHT_Q_MULT if gate else GATE_LOOSE_Q_MULT
            q = max(
                1, min(MAX_Q, int(max(GATE_LOOSE_Q_MIN, round(q0 * mult))))
            )  # new-entry sizing; exits recompute q below
            q = min(q, lim - pos, lim + pos)
            if q <= 0:
                continue

            bv, av = _top_book_size(d)
            top_sum = bv + av

            if abs(resid) < RESID_EXIT and abs(pos) > 0:
                qf = min(MAX_Q, max(1, int(20.0 / max(veg, 1e-6))))
                qf = min(qf, lim - pos, lim + pos)
                if pos > 0 and qf > 0:
                    orders[sym].append(Order(sym, ba, -min(qf, pos)))
                elif pos < 0 and qf > 0:
                    orders[sym].append(Order(sym, bb, min(qf, -pos)))
            elif res_ema > RESID_ENTER and pos > -lim + q and top_sum >= MIN_TOP_SUM_VOL:
                p = _entry_sell_price(bb, ba, dS, stress, sym)
                orders[sym].append(Order(sym, p, -q))
            elif res_ema < -RESID_ENTER and pos < lim - q and top_sum >= MIN_TOP_SUM_VOL:
                p = _entry_buy_price(bb, ba, dS, stress, sym)
                orders[sym].append(Order(sym, p, q))

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
