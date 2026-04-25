
"""
Family 12 v16: v15 + burst momentum bias on theo (signed extract move).

Tape lead: after extract |z| shocks, voucher mid diffs continue in the same direction
as the extract move at high rates (see analysis_outputs/shock_continuation_by_strike_day.csv).
In burst, shift fair theo along the move (scaled by response) before edge/lot scaling.
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

def _symbol_for_product(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
UNDERLYING = "VELVETFRUIT_EXTRACT"
EMA_KEY = "fam12_spline_momentum_bias"

Z_WINDOW = 50
Z_SPIKE = 2.2
ABS_DS_SPIKE = 7.0
WARMUP_DIV100 = 10
ATM_REL_BURST = 0.24
ATM_REL_CALM = 0.10
BURST_MAX_LOT = 80

# quoting edges around theo
EDGE_CALM = 0.9
EDGE_BURST = 1.6

# strike response proxy from analysis (mean beta_dV_dS by strike)
BETA_PROXY = {
    4000: 0.7451, 4500: 0.6618, 5000: 0.6535, 5100: 0.5772,
    5200: 0.4366, 5300: 0.2727, 5400: 0.1290, 5500: 0.0550,
    6000: 0.0010, 6500: 0.0010,
}

# burst fair shift along extract move (price units), scaled by resp
MOM_BIAS = 0.55


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
    # Light-weight cubic polynomial fallback in log-k/s space (family style without scipy dep)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    deg = 3 if len(x) >= 4 else (2 if len(x) >= 3 else 1)
    coef = np.polyfit(x, y, deg)
    return coef.tolist(), [float(x.min()), float(x.max())]


def _eval_spline(coef: list[float], x: float) -> float:
    return float(np.polyval(np.asarray(coef, dtype=float), float(x)))


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

        dS_step = (S - hist[-1]) if hist else 0.0
        abs_dS = abs(dS_step)
        hist.append(S)
        if len(hist) >= 2 and hist[-2] > 0:
            dlog_hist.append(math.log(hist[-1] / hist[-2]))
        hist = hist[-120:]
        dlog_hist = dlog_hist[-200:]

        z = _rolling_z(dlog_hist, Z_WINDOW)
        burst = (z >= Z_SPIKE) or (abs_dS >= ABS_DS_SPIKE)
        move_sign = 0
        if dS_step > 0.25:
            move_sign = 1
        elif dS_step < -0.25:
            move_sign = -1
        atm_rel = ATM_REL_BURST if burst else ATM_REL_CALM
        lot_cap = BURST_MAX_LOT if burst else 10_000

        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < WARMUP_DIV100:
            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        csv_day = int(getattr(state, "csv_day", 0))
        T = t_years(csv_day, ts)

        # collect IV points from all strikes in cluster (ungated)
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

        if len(xs) < 3:
            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        coef, xbounds = _fit_cubic_iv_spline(xs, ys)
        x_lo, x_hi = xbounds

        pos = getattr(state, "position", {}) or {}
        out: dict[str, list[Order]] = {}
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

            # delta/beta scaling from underlying-response analysis:
            # lower response (small |delta| and beta proxy) -> wider edge + smaller lot.
            beta_w = float(BETA_PROXY.get(int(k), 0.1))
            delta_w = float(abs(delta))
            resp = max(0.05, min(1.0, 0.55 * delta_w + 0.45 * beta_w))
            eff_edge = edge / resp
            eff_lot_cap = max(8, int(lot_cap * resp))

            theo_exec = theo
            if burst and move_sign != 0:
                theo_exec = theo + float(move_sign) * MOM_BIAS * (0.5 + 0.5 * resp)

            buy_cap = min(lim - pos_k, eff_lot_cap)
            sell_cap = min(lim + pos_k, eff_lot_cap)

            # quote around theo with scaled edge; take if book crosses fair+edge.
            bid_px = int(math.floor(theo_exec - eff_edge))
            ask_px = int(math.ceil(theo_exec + eff_edge))
            orders: list[Order] = []

            # take mispriced asks
            if ba <= bid_px and buy_cap > 0:
                q = max(0, min(buy_cap, abs(int(getattr(depths[sym], 'sell_orders', {}).get(ba, 0)))))
                if q > 0:
                    orders.append(Order(sym, int(ba), int(q)))
                    buy_cap -= q
            # take mispriced bids
            if bb >= ask_px and sell_cap > 0:
                q = max(0, min(sell_cap, abs(int(getattr(depths[sym], 'buy_orders', {}).get(bb, 0)))))
                if q > 0:
                    orders.append(Order(sym, int(bb), -int(q)))
                    sell_cap -= q

            # passive quotes
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
        return out, 0, json.dumps(store, separators=(",", ":"))
