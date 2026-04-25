
"""
Family 12 variation B (liquidity-gated): same smile cubic spline log-k setup,
with strike inclusion gated by per-book spread/depth quality.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

_REPO = Path(__file__).resolve().parents[3]

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
UNDERLYING = "VELVETFRUIT_EXTRACT"

Z_WINDOW = 50
Z_SPIKE = 2.2
ABS_DS_SPIKE = 7.0
WARMUP_DIV100 = 10
ATM_REL_BURST = 0.24
ATM_REL_CALM = 0.10
BURST_MAX_LOT = 80

EDGE_CALM = 0.9
EDGE_BURST = 1.6

# liquidity quality gates (from analysis_outputs/liquidity_quality_by_strike_day.csv)
MAX_SPREAD = 6.5
MIN_DEPTH = 26.0
MIN_QUALITY = 4.0


def _symbol_for_product(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def t_years(csv_day: int, ts: int) -> float:
    dte_open = 8 - int(csv_day)
    prog = (int(ts) // 100) / 10_000.0
    return max((dte_open - prog) / 365.0, 1e-6)


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _rolling_z(series: list[float], window: int) -> float:
    if len(series) < max(window, 5):
        return 0.0
    arr = np.asarray(series[-window:], dtype=float)
    return float(abs(arr[-1] - arr.mean()) / (arr.std() + 1e-9))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call(S: float, K: float, T: float, sigma: float) -> tuple[float, float, float]:
    if T <= 0 or sigma <= 1e-9:
        intrinsic = max(S - K, 0.0)
        delta = 1.0 if S > K else 0.0
        return intrinsic, delta, 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / v
    d2 = d1 - v
    price = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    vega = S * _norm_pdf(d1) * math.sqrt(T)
    return float(price), float(delta), float(vega)


def implied_vol_call(market: float, S: float, K: float, T: float) -> float:
    intrinsic = max(S - K, 0.0)
    if not (S > 0 and K > 0 and T > 0):
        return float("nan")
    if market <= intrinsic + 1e-9 or market >= S - 1e-9:
        return float("nan")
    lo, hi = 1e-4, 5.0
    flo = bs_call(S, K, T, lo)[0] - market
    fhi = bs_call(S, K, T, hi)[0] - market
    if flo * fhi > 0:
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = bs_call(S, K, T, mid)[0] - market
        if abs(fm) < 1e-7:
            return float(mid)
        if flo * fm <= 0:
            hi = mid
            fhi = fm
        else:
            lo = mid
            flo = fm
    return float(0.5 * (lo + hi))


def _fit_cubic_iv_spline(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    deg = 3 if len(x) >= 4 else (2 if len(x) >= 3 else 1)
    coef = np.polyfit(x, y, deg)
    return coef.tolist(), [float(x.min()), float(x.max())]


def _eval_spline(coef: list[float], x: float) -> float:
    return float(np.polyval(np.asarray(coef, dtype=float), float(x)))


def _passes_liquidity_gate(bb: int, ba: int, bv: int, av: int) -> bool:
    spread = float(ba - bb)
    depth = float(max(0, bv) + max(0, av))
    quality = depth / max(spread, 1e-9)
    return (spread <= MAX_SPREAD) and (depth >= MIN_DEPTH) and (quality >= MIN_QUALITY)


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        hist = store.get("s_hist")
        if not isinstance(hist, list):
            hist = []
        hist = [float(x) for x in hist if isinstance(x, (int, float))][-120:]

        dlog_hist = store.get("dlog_hist")
        if not isinstance(dlog_hist, list):
            dlog_hist = []
        dlog_hist = [float(x) for x in dlog_hist if isinstance(x, (int, float))][-200:]

        depths = getattr(state, "order_depths", {}) or {}
        sym_u = _symbol_for_product(state, UNDERLYING)
        if sym_u is None or sym_u not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        depth_u = depths[sym_u]
        buys_u = getattr(depth_u, "buy_orders", {}) or {}
        sells_u = getattr(depth_u, "sell_orders", {}) or {}
        if not buys_u or not sells_u:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ubb = max(buys_u.keys())
        uba = min(sells_u.keys())
        S = 0.5 * (float(ubb) + float(uba))

        abs_dS = abs(S - hist[-1]) if hist else 0.0
        hist.append(S)
        if len(hist) >= 2 and hist[-2] > 0:
            dlog_hist.append(math.log(hist[-1] / hist[-2]))
        hist = hist[-120:]
        dlog_hist = dlog_hist[-200:]

        z = _rolling_z(dlog_hist, Z_WINDOW)
        burst = (z >= Z_SPIKE) or (abs_dS >= ABS_DS_SPIKE)
        atm_rel = ATM_REL_BURST if burst else ATM_REL_CALM
        lot_cap = BURST_MAX_LOT if burst else 10_000

        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < WARMUP_DIV100:
            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        csv_day = int(getattr(state, "csv_day", 0))
        T = t_years(csv_day, ts)

        xs, ys = [], []
        sym_by_k: dict[int, str] = {}
        book_by_k: dict[int, tuple[int, int, int, int]] = {}
        gated_out = 0
        for k in STRIKES:
            if abs(k / S - 1.0) > atm_rel:
                continue
            p = f"VEV_{k}"
            sym = _symbol_for_product(state, p)
            if sym is None or sym not in depths:
                continue
            d = depths[sym]
            buys = getattr(d, "buy_orders", {}) or {}
            sells = getattr(d, "sell_orders", {}) or {}
            if not buys or not sells:
                continue
            bb = max(buys.keys())
            ba = min(sells.keys())
            bv = abs(int(buys.get(bb, 0)))
            av = abs(int(sells.get(ba, 0)))
            if not _passes_liquidity_gate(int(bb), int(ba), int(bv), int(av)):
                gated_out += 1
                continue
            m = 0.5 * (float(bb) + float(ba))
            iv = implied_vol_call(m, S, float(k), T)
            if not (math.isfinite(iv) and 0.01 <= iv <= 3.0):
                continue
            x = math.log(float(k) / S)
            xs.append(x)
            ys.append(iv)
            sym_by_k[k] = sym
            book_by_k[k] = (int(bb), int(ba), int(bv), int(av))

        if len(xs) < 3:
            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            store["gated_out_last"] = int(gated_out)
            return {}, 0, json.dumps(store, separators=(",", ":"))

        coef, xbounds = _fit_cubic_iv_spline(xs, ys)
        x_lo, x_hi = xbounds

        pos = getattr(state, "position", {}) or {}
        out: dict[str, list[Order]] = {}
        edge = EDGE_BURST if burst else EDGE_CALM

        for k, sym in sym_by_k.items():
            bb, ba, bv, av = book_by_k[k]
            x = math.log(float(k) / S)
            x = min(max(x, x_lo), x_hi)
            iv_hat = max(0.01, min(3.0, _eval_spline(coef, x)))
            theo, delta, vega = bs_call(S, float(k), T, iv_hat)
            pos_k = int(pos.get(sym, 0))
            lim = 300
            buy_cap = min(lim - pos_k, lot_cap)
            sell_cap = min(lim + pos_k, lot_cap)

            bid_px = int(math.floor(theo - edge))
            ask_px = int(math.ceil(theo + edge))
            orders: list[Order] = []

            if ba <= bid_px and buy_cap > 0:
                q = max(0, min(buy_cap, av))
                if q > 0:
                    orders.append(Order(sym, int(ba), int(q)))
                    buy_cap -= q
            if bb >= ask_px and sell_cap > 0:
                q = max(0, min(sell_cap, bv))
                if q > 0:
                    orders.append(Order(sym, int(bb), -int(q)))
                    sell_cap -= q

            if buy_cap > 0:
                px = min(int(bb + 1), bid_px)
                orders.append(Order(sym, int(px), int(min(buy_cap, lot_cap))))
            if sell_cap > 0:
                px = max(int(ba - 1), ask_px)
                orders.append(Order(sym, int(px), -int(min(sell_cap, lot_cap))))

            if orders:
                out[sym] = orders

        store["s_hist"] = hist
        store["dlog_hist"] = dlog_hist
        store["last_burst"] = bool(burst)
        store["fit_coef"] = coef
        store["gated_out_last"] = int(gated_out)
        return out, 0, json.dumps(store, separators=(",", ":"))
