
"""
Family 12 v22: v15 smile MM, gated on joint 5200/5300 tight books only (no extract).

v21's naive extract mean reversion bled on worse fills; this variant keeps the risk
on/off filter from round3work/vouchers_final_strategy/STRATEGY.txt but trades VEV
options only. No HYDROGEL.
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

# Joint gate: STRATEGY.txt, Sonic / inclineGod
SPREAD_TH = 2
GATE_5200 = 5200
GATE_5300 = 5300

Z_WINDOW = 50
Z_SPIKE = 2.2
ABS_DS_SPIKE = 7.0
WARMUP_DIV100 = 10
ATM_REL_BURST = 0.24
ATM_REL_CALM = 0.10
BURST_MAX_LOT = 80
EDGE_CALM = 0.9
EDGE_BURST = 1.6

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
UNDERLYING = "VELVETFRUIT_EXTRACT"

BETA_PROXY = {
    4000: 0.7451, 4500: 0.6618, 5000: 0.6535, 5100: 0.5772,
    5200: 0.4366, 5300: 0.2727, 5400: 0.1290, 5500: 0.0550,
    6000: 0.0010, 6500: 0.0010,
}


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


def _l1_spread_and_ok(depths: dict[str, Any], sym: str | None) -> tuple[float, bool]:
    if sym is None or sym not in depths:
        return float("inf"), False
    d = depths[sym]
    buys = getattr(d, "buy_orders", {}) or {}
    sells = getattr(d, "sell_orders", {}) or {}
    if not buys or not sells:
        return float("inf"), False
    bb = max(buys.keys())
    ba = min(sells.keys())
    return float(ba) - float(bb), True


def _joint_tight_gate(depths: dict[str, Any], k5200_sym: str | None, k5300_sym: str | None) -> bool:
    s5200, ok1 = _l1_spread_and_ok(depths, k5200_sym)
    s5300, ok2 = _l1_spread_and_ok(depths, k5300_sym)
    if not (ok1 and ok2):
        return False
    return s5200 <= float(SPREAD_TH) and s5300 <= float(SPREAD_TH)


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

        k5200_s = _symbol_for_product(state, f"VEV_{GATE_5200}")
        k5300_s = _symbol_for_product(state, f"VEV_{GATE_5300}")

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

        joint_tight = _joint_tight_gate(depths, k5200_s, k5300_s)
        s5200, _ = _l1_spread_and_ok(depths, k5200_s)
        s5300, _ = _l1_spread_and_ok(depths, k5300_s)
        store["joint_tight"] = bool(joint_tight)
        store["s5200_spread"] = float(s5200) if math.isfinite(s5200) else -1.0
        store["s5300_spread"] = float(s5300) if math.isfinite(s5300) else -1.0

        if not joint_tight:
            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            store["last_burst"] = bool(burst)
            return {}, 0, json.dumps(store, separators=(",", ":"))

        csv_day = int(getattr(state, "csv_day", 0))
        T = t_years(csv_day, ts)
        pos = getattr(state, "position", {}) or {}

        xs, ys = [], []
        mid_by_k: dict[int, float] = {}
        sym_by_k: dict[int, str] = {}
        book_by_k: dict[int, tuple[int, int]] = {}
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
            m = 0.5 * (float(bb) + float(ba))
            iv = implied_vol_call(m, S, float(k), T)
            if not (math.isfinite(iv) and 0.01 <= iv <= 3.0):
                continue
            x = math.log(float(k) / S)
            xs.append(x)
            ys.append(iv)
            mid_by_k[k] = m
            sym_by_k[k] = sym
            book_by_k[k] = (int(bb), int(ba))

        out: dict[str, list[Order]] = {}
        if len(xs) < 3:
            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            store["last_burst"] = bool(burst)
        else:
            coef, xbounds = _fit_cubic_iv_spline(xs, ys)
            x_lo, x_hi = xbounds

            edge = EDGE_BURST if burst else EDGE_CALM

            for k, m in mid_by_k.items():
                x = math.log(float(k) / S)
                x = min(max(x, x_lo), x_hi)
                iv_hat = max(0.01, min(3.0, _eval_spline(coef, x)))
                theo, delta, vega = bs_call(S, float(k), T, iv_hat)
                bb, ba = book_by_k[k]
                sym = sym_by_k[k]
                pos_k = int(pos.get(sym, 0))
                lim = 300

                beta_w = float(BETA_PROXY.get(int(k), 0.1))
                delta_w = float(abs(delta))
                resp = max(0.05, min(1.0, 0.55 * delta_w + 0.45 * beta_w))
                eff_edge = edge / resp
                eff_lot_cap = max(8, int(lot_cap * resp))

                buy_cap = min(lim - pos_k, eff_lot_cap)
                sell_cap = min(lim + pos_k, eff_lot_cap)

                bid_px = int(math.floor(theo - eff_edge))
                ask_px = int(math.ceil(theo + eff_edge))
                orders: list[Order] = []

                if ba <= bid_px and buy_cap > 0:
                    q = max(0, min(buy_cap, abs(int(getattr(depths[sym], "sell_orders", {}).get(ba, 0)))))
                    if q > 0:
                        orders.append(Order(sym, int(ba), int(q)))
                        buy_cap -= q
                if bb >= ask_px and sell_cap > 0:
                    q = max(0, min(sell_cap, abs(int(getattr(depths[sym], "buy_orders", {}).get(bb, 0)))))
                    if q > 0:
                        orders.append(Order(sym, int(bb), -int(q)))
                        sell_cap -= q

                if buy_cap > 0:
                    px = min(int(bb + 1), bid_px)
                    orders.append(Order(sym, int(px), int(min(buy_cap, eff_lot_cap))))
                if sell_cap > 0:
                    px = max(int(ba - 1), ask_px)
                    orders.append(Order(sym, int(px), -int(min(sell_cap, eff_lot_cap))))

                if orders:
                    out[sym] = orders

            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            store["last_burst"] = bool(burst)
            store["fit_coef"] = coef

        if len(xs) < 3:
            return out, 0, json.dumps(store, separators=(",", ":"))
        return out, 0, json.dumps(store, separators=(",", ":"))
