"""
Round 4 iteration 5 — Phase 2 deferred: IV smile steepness filter × Phase 1/3 signal.

Offline calibration (r4_phase2b_iv_smile_mark22.py on Mark 22 voucher-sell timestamps, days 1–3):
quadratic IV smile coefficient a (IV ~ a*m^2 + b*m + c, m=log(K/S)/sqrt(T)) has pooled q75 ≈ 0.08866.

trader_v5 = trader_v3 logic (Sonic gate + Mark 67 aggressive buy extract + lift ask) only when
current smile steepness a >= ROUND4_STEEP_MIN (top ~quartile of that reference distribution).
If fewer than 5 strikes produce finite IV, skip (no order).
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH_SPREAD = 2
BUY_Q = 24
EX_LIM = 200
COOLDOWN = 6
WARMUP = 5
ROUND4_STEEP_MIN = 0.08866

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHERS = [f"VEV_{k}" for k in STRIKES]

_EMA_KEY = "ema_S"
EMA_N = 12
_LAST_FIRE = "last_fire_bucket"


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _sym(state: TradingState, prod: str) -> str | None:
    for s, lst in (getattr(state, "listings", {}) or {}).items():
        if getattr(lst, "product", None) == prod:
            return s
    return None


def _ba(d: OrderDepth | None) -> tuple[int | None, int | None]:
    if d is None:
        return None, None
    b, s = getattr(d, "buy_orders", None) or {}, getattr(d, "sell_orders", None) or {}
    if not b or not s:
        return None, None
    return max(b.keys()), min(s.keys())


def _mid_depth(d: OrderDepth | None) -> float | None:
    bb, ba = _ba(d)
    if bb is None or ba is None:
        return None
    return 0.5 * (float(bb) + float(ba))


def _ema(p: float | None, x: float, n: int) -> float:
    if p is None:
        return x
    a = 2.0 / (n + 1.0)
    return a * x + (1.0 - a) * p


def _day(td: dict[str, Any], ts: int, s: float) -> int:
    if ts != 0:
        return int(td.get("csv_day", 0))
    h = td.get("open_S_hist")
    if not isinstance(h, list):
        h = []
    c = round(float(s), 2)
    if not h or abs(float(h[-1]) - c) > 0.25:
        h.append(c)
    td["open_S_hist"] = h[:4]
    return max(0, min(len(h) - 1, 2))


def _sonic_tight(depths: dict[str, OrderDepth], s5: str, s3: str) -> bool:
    b5, a5 = _ba(depths.get(s5))
    b3, a3 = _ba(depths.get(s3))
    if None in (b5, a5, b3, a3):
        return False
    return (a5 - b5) <= TH_SPREAD and (a3 - b3) <= TH_SPREAD


def _m67_aggressive_buy_extract(state: TradingState, sym_ex: str, ask: int) -> bool:
    m = getattr(state, "market_trades", None) or {}
    for tr in m.get(sym_ex, []) or []:
        if getattr(tr, "buyer", None) == "Mark 67" and int(getattr(tr, "price", 0)) >= int(ask):
            return True
    return False


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _iv_call(mid: float, S: float, K: float, T: float, r: float = 0.0) -> float:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-9:
        return float("nan")
    if mid >= S - 1e-9:
        return float("nan")
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    def f(sig: float) -> float:
        return _bs_call(S, K, T, sig, r) - mid

    lo, hi = 1e-5, 15.0
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    for _ in range(50):
        m = 0.5 * (lo + hi)
        fm = f(m)
        if fm > 0:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def _t_years(csv_day: int) -> float:
    return max(1e-6, float(5 - int(csv_day)) / 365.0)


def _smile_steepness(S: float, T: float, depths: dict[str, OrderDepth], state: TradingState) -> float:
    ks: list[float] = []
    ivs: list[float] = []
    sqrtT = math.sqrt(T)
    for prod, K in zip(VOUCHERS, [float(k) for k in STRIKES]):
        sym = _sym(state, prod)
        if sym is None:
            continue
        mid = _mid_depth(depths.get(sym))
        if mid is None:
            continue
        iv = _iv_call(mid, S, K, T, 0.0)
        if iv == iv and iv > 0:
            ks.append(K)
            ivs.append(iv)
    if len(ivs) < 5:
        return float("nan")
    m_arr = np.array([math.log(k / S) / sqrtT for k in ks], dtype=float)
    y_arr = np.array(ivs, dtype=float)
    try:
        coeff = np.polyfit(m_arr, y_arr, 2)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")
    if not np.all(np.isfinite(coeff)):
        return float("nan")
    return float(coeff[0])


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        sym_ex = _sym(state, "VELVETFRUIT_EXTRACT")
        s520 = _sym(state, "VEV_5200")
        s530 = _sym(state, "VEV_5300")
        if not sym_ex or not s520 or not s530:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        du = depths.get(sym_ex)
        ubb, uba = _ba(du)
        if ubb is None or uba is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        s_raw = 0.5 * (ubb + uba)
        pe = td.get(_EMA_KEY)
        td[_EMA_KEY] = _ema(float(pe) if isinstance(pe, (int, float)) else None, s_raw, EMA_N)
        csv_day = _day(td, ts, s_raw)
        td["csv_day"] = csv_day

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _sonic_tight(depths, s520, s530):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _m67_aggressive_buy_extract(state, sym_ex, int(uba)):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        steep = _smile_steepness(s_raw, _t_years(csv_day), depths, state)
        if steep != steep or steep < ROUND4_STEEP_MIN:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        bucket = ts // 100
        last = td.get(_LAST_FIRE)
        if isinstance(last, int) and bucket - last < COOLDOWN:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        pos_e = int(pos.get(sym_ex, 0))
        qb = min(BUY_Q, EX_LIM - pos_e)
        if qb <= 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        td[_LAST_FIRE] = int(bucket)
        return {sym_ex: [Order(sym_ex, int(uba), qb)]}, 0, json.dumps(td, separators=(",", ":"))
