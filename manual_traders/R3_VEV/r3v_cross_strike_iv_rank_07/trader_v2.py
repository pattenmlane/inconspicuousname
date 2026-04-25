"""
Round 3 — cross-strike IV rank fade (r3v_cross_strike_iv_rank_07), iteration v2.

Grid on v1: NEIGHBOR_IV_CAP=0.15 (tape sweep shows high-decile pairs rarely pass tight caps
when gaps are measured vs adjacent strikes in sorted-strike order; loosen further),
BASE_Q=5 to reduce adverse selection on wide-spread wings.

No extract hedge (same as v1). Timing unchanged.
"""
from __future__ import annotations

import json
import math
from typing import Any

from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState
from scipy.stats import norm

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]

# --- strategy parameters (v2: cap + size sweep) ---
NEIGHBOR_IV_CAP = 0.15
BASE_Q = 5
MAX_CLUSTER_Q = 22
WARMUP_STEPS = 8
# Order / IV work only every N centisecond-buckets (ts advances by 100); keeps backtests tractable.
ACTION_STRIDE = 4
EMA_S_N = 12
_EMA_KEY = "ema_S"


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
    """Infer historical CSV day 0/1/2 from distinct session-open underlying mids."""
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
            bids.append(int(bb))
            asks.append(int(ba))
            syms.append(sym)

        n = len(ivs)
        if n < 8:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        order_idx = sorted(range(n), key=lambda i: ivs[i])
        low_pair = set(order_idx[:2])
        high_pair = set(order_idx[-2:])

        def neighbor_iv_gap(i: int) -> float:
            """Max |IV diff| to adjacent strike in this tick's valid subset (sorted-by-strike order)."""
            gaps = []
            if i > 0:
                gaps.append(abs(ivs[i] - ivs[i - 1]))
            if i < n - 1:
                gaps.append(abs(ivs[i] - ivs[i + 1]))
            return max(gaps) if gaps else 0.0

        orders_by_sym: dict[str, list[Order]] = {}

        def add_o(sym: str, price: int, qty: int) -> None:
            if qty == 0:
                return
            orders_by_sym.setdefault(sym, []).append(Order(sym, int(price), int(qty)))

        for i in range(n):
            sym = syms[i]
            pos_i = int(pos.get(sym, 0))
            lim = 300
            bb2, ba2 = bids[i], asks[i]
            if neighbor_iv_gap(i) > NEIGHBOR_IV_CAP:
                continue
            q_use = BASE_Q
            if i > 0 and ((i - 1) in low_pair or (i - 1) in high_pair):
                q_use = min(MAX_CLUSTER_Q, BASE_Q * 2)
            if i < n - 1 and ((i + 1) in low_pair or (i + 1) in high_pair):
                q_use = min(MAX_CLUSTER_Q, max(q_use, BASE_Q * 2))

            if i in low_pair:
                qb = min(q_use, lim - pos_i)
                if qb > 0 and bb2 + 1 < ba2:
                    add_o(sym, bb2 + 1, qb)
            if i in high_pair:
                qs = min(q_use, lim + pos_i)
                if qs > 0 and ba2 - 1 > bb2:
                    add_o(sym, ba2 - 1, -qs)

        return orders_by_sym, 0, json.dumps(td, separators=(",", ":"))
