"""
Round 3 — smile params updated slowly (EWMA) so theoretical fair does not jump every tick.

Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_* only.

DTE / T: round3work/round3description.txt (CSV historical day index vs TTE) plus intraday
winding as in round3work/plotting/original_method/combined_analysis/plot_iv_smile_round3.py
(dte_effective, t_years_effective).

Iteration 0 params (small grid later): EWMA on quadratic IV-in-m_t coeffs; quote vouchers
off BS(fair_iv); slow hydrogel fair from mid.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from datamodel import Order, OrderDepth, TradingState

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"

LIMITS = {H: 200, U: 200, **{v: 300 for v in VOUCHERS}}

# --- Sticky smile (iteration 0) ---
EWMA_ALPHA = 0.03
HYD_EWMA_ALPHA = 0.004
MIN_IV_POINTS = 6
QUOTE_SIZE_VEV = 12
QUOTE_SIZE_DELTA = 10
MM_WIDTH_VEV = 2
EDGE_CLIP = 40.0
VEGA_SCALE_REF = 80.0

# Hydrogel MM off slow fair
HYD_MM_WIDTH = 3
HYD_QUOTE_SIZE = 8


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years_effective(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12 or S <= 0 or K <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return S * norm.pdf(d1) * math.sqrt(T)


def implied_vol_call(market: float, S: float, K: float, T: float, r: float = 0.0) -> float | None:
    intrinsic = max(S - K, 0.0)
    if market <= intrinsic + 1e-6:
        return None
    if market >= S - 1e-6:
        return None
    if S <= 0 or K <= 0 or T <= 0:
        return None

    def f(sig: float) -> float:
        return bs_call_price(S, K, T, sig, r) - market

    lo, hi = 1e-5, 12.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-7, rtol=1e-7))
    except ValueError:
        return None


def best_bid_ask(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders.keys()), min(depth.sell_orders.keys())


def micro_mid(depth: OrderDepth) -> float | None:
    bb, ba = best_bid_ask(depth)
    if bb is None or ba is None:
        return None
    return 0.5 * (bb + ba)


def infer_csv_day(s_mid: float, h_mid: float | None) -> int:
    """Map backtest day_num (0,1,2) to DTE calendar using extract + hydrogel mids at session open."""
    if abs(s_mid - 5250.0) < 1.5 and h_mid is not None and abs(h_mid - 10000.0) < 2.5:
        return 0
    if abs(s_mid - 5245.0) < 1.5 and h_mid is not None and abs(h_mid - 9958.0) < 2.5:
        return 1
    if abs(s_mid - 5267.5) < 2.0:
        return 2
    return 0


class Trader:
    def run(self, state: TradingState):
        store: dict[str, Any] = {}
        raw = getattr(state, "traderData", "") or ""
        if raw:
            try:
                o = json.loads(raw)
                if isinstance(o, dict):
                    store = o
            except (json.JSONDecodeError, TypeError):
                store = {}

        ts = int(getattr(state, "timestamp", 0))
        pos = getattr(state, "position", {}) or {}
        depths = getattr(state, "order_depths", {}) or {}

        def orders_for(sym: str) -> list[Order]:
            return []

        out: dict[str, list[Order]] = {}

        # --- Underlying + hydrogel mids ---
        du = depths.get(U)
        dh = depths.get(H)
        if du is None:
            return out, 0, json.dumps(store, separators=(",", ":"))

        s_mid = micro_mid(du)
        if s_mid is None:
            return out, 0, json.dumps(store, separators=(",", ":"))

        h_mid = micro_mid(dh) if dh is not None else None
        csv_day = int(store.get("csv_day", -1))
        if csv_day < 0 or ts == 0:
            csv_day = infer_csv_day(float(s_mid), float(h_mid) if h_mid is not None else None)
            store["csv_day"] = csv_day

        T = t_years_effective(csv_day, ts)
        sqrtT = math.sqrt(T) if T > 0 else 1e-6

        # --- Fit instantaneous smile IV(m_t), EWMA coeffs ---
        xs: list[float] = []
        ys: list[float] = []
        vegas: list[float] = []
        mids_v: dict[str, float] = {}

        for v in VOUCHERS:
            dv = depths.get(v)
            if dv is None:
                continue
            m = micro_mid(dv)
            if m is None:
                continue
            K = float(v.split("_")[1])
            iv = implied_vol_call(float(m), float(s_mid), K, T, 0.0)
            if iv is None:
                continue
            m_t = math.log(K / float(s_mid)) / sqrtT
            xs.append(m_t)
            ys.append(iv)
            vegas.append(bs_vega(float(s_mid), K, T, iv, 0.0))
            mids_v[v] = float(m)

        coeff_raw: list[float] | None = None
        if len(xs) >= MIN_IV_POINTS:
            xf = np.asarray(xs, dtype=float)
            yf = np.asarray(ys, dtype=float)
            wf = np.asarray(vegas, dtype=float) + 1e-6
            coeff_raw = list(np.polyfit(xf, yf, 2, w=wf))

        ema = store.get("ema_coeff")
        if not isinstance(ema, list) or len(ema) != 3:
            ema = [0.15, 0.0, 0.24]
        if coeff_raw is not None:
            a = EWMA_ALPHA
            ema = [(1 - a) * float(ema[i]) + a * float(coeff_raw[i]) for i in range(3)]
        store["ema_coeff"] = ema
        c2, c1, c0 = float(ema[0]), float(ema[1]), float(ema[2])

        # --- VEV quotes from sticky smile ---
        for v in VOUCHERS:
            if v not in mids_v:
                continue
            dv = depths.get(v)
            if dv is None:
                continue
            K = float(v.split("_")[1])
            m_t = math.log(K / float(s_mid)) / sqrtT
            fair_iv = max(1e-4, min(8.0, float(np.polyval([c2, c1, c0], m_t))))
            fair = bs_call_price(float(s_mid), K, T, fair_iv, 0.0)
            mid = mids_v[v]
            vega = bs_vega(float(s_mid), K, T, fair_iv, 0.0)
            edge = mid - fair
            w = max(0.35, min(2.5, vega / VEGA_SCALE_REF))
            shift = MM_WIDTH_VEV + w * max(-EDGE_CLIP, min(EDGE_CLIP, edge)) * 0.06

            pos_v = int(pos.get(v, 0))
            lim = LIMITS[v]
            bb, ba = best_bid_ask(dv)
            if bb is None or ba is None:
                continue

            bid_px = int(round(fair - shift))
            ask_px = int(round(fair + shift))
            bid_px = min(bid_px, ba - 1)
            ask_px = max(ask_px, bb + 1)
            if ask_px <= bid_px:
                ask_px = bid_px + 1

            q_buy = min(QUOTE_SIZE_VEV, lim - pos_v)
            q_sell = min(QUOTE_SIZE_VEV, lim + pos_v)
            if edge > 2.0:
                q_buy = min(q_buy + QUOTE_SIZE_DELTA, lim - pos_v)
            if edge < -2.0:
                q_sell = min(q_sell + QUOTE_SIZE_DELTA, lim + pos_v)

            ol: list[Order] = []
            if q_buy > 0 and bid_px > 0:
                ol.append(Order(v, bid_px, q_buy))
            if q_sell > 0:
                ol.append(Order(v, ask_px, -q_sell))
            if ol:
                out[v] = ol

        # --- Extract: lean inventory with delta from smile ---
        pos_u = int(pos.get(U, 0))
        delta_hat = 0.0
        for v in VOUCHERS:
            if v not in mids_v:
                continue
            K = float(v.split("_")[1])
            iv = max(1e-4, float(np.polyval([c2, c1, c0], math.log(K / float(s_mid)) / sqrtT)))
            v_ = iv * math.sqrt(T)
            if v_ <= 1e-12:
                continue
            d1 = (math.log(float(s_mid) / K) + 0.5 * iv * iv * T) / v_
            delta_hat += float(pos.get(v, 0)) * float(norm.cdf(d1))

        bias = max(-1.0, min(1.0, -delta_hat / 450.0))
        u_fair = float(s_mid) + 1.8 * bias
        u_bid = int(round(u_fair - 2))
        u_ask = int(round(u_fair + 2))
        bb_u, ba_u = best_bid_ask(du)
        if bb_u is not None and ba_u is not None:
            u_bid = min(u_bid, ba_u - 1)
            u_ask = max(u_ask, bb_u + 1)
            if u_ask <= u_bid:
                u_ask = u_bid + 1
            qb = min(QUOTE_SIZE_VEV, 200 - pos_u)
            qs = min(QUOTE_SIZE_VEV, 200 + pos_u)
            uo: list[Order] = []
            if qb > 0:
                uo.append(Order(U, u_bid, qb))
            if qs > 0:
                uo.append(Order(U, u_ask, -qs))
            if uo:
                out[U] = uo

        # --- Hydrogel: slow EWMA fair, wide MM ---
        if dh is not None and h_mid is not None:
            hf = store.get("hyd_fair")
            if not isinstance(hf, (int, float)) or not math.isfinite(float(hf)):
                hf = float(h_mid)
            hf = (1.0 - HYD_EWMA_ALPHA) * float(hf) + HYD_EWMA_ALPHA * float(h_mid)
            store["hyd_fair"] = hf
            pos_h = int(pos.get(H, 0))
            hb = int(round(hf - HYD_MM_WIDTH))
            ha = int(round(hf + HYD_MM_WIDTH))
            bb_h, ba_h = best_bid_ask(dh)
            if bb_h is not None and ba_h is not None:
                hb = min(hb, ba_h - 1)
                ha = max(ha, bb_h + 1)
                if ha <= hb:
                    ha = hb + 1
                qh = min(HYD_QUOTE_SIZE, 200 - pos_h)
                qhs = min(HYD_QUOTE_SIZE, 200 + pos_h)
                ho: list[Order] = []
                if qh > 0:
                    ho.append(Order(H, hb, qh))
                if qhs > 0:
                    ho.append(Order(H, ha, -qhs))
                if ho:
                    out[H] = ho

        return out, 0, json.dumps(store, separators=(",", ":"))
