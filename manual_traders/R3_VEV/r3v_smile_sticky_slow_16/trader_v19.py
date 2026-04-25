"""
Round 3 — family 12: v15 + extract-shock core-only passive size (no taker).

- Identical fair/smile/width to trader_v15 (spline + poly blend w=0.35, sticky nodes, wing filter).
- Persist prior VELVETFRUIT_EXTRACT mid; when |S−S_prev|>=2, add CORE_SHOCK_SIZE_BOOST to both sides’ clip
  size for strikes 5000-5300 only. Tape (extract_shock_core_mid_asymmetry.json): on shocks, core median |dm|
  is 2 vs 0.5 for 5400/5500 and 0 for deep wings — size bump is core-only to avoid wing adverse flow.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.stats import norm

from datamodel import Order, OrderDepth, TradingState

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"

LIMITS = {H: 200, U: 200, **{v: 300 for v in VOUCHERS}}

NODE_EWMA_ALPHA = 0.08
POLY_EWMA_ALPHA = 0.045
# Weight on quadratic BS fair in the blended fair price; spline weight is (1 - this).
FAIR_BLEND_POLY_W = 0.35
WING_REG_LAMBDA = 0.1
HYD_EWMA_ALPHA = 0.004
MIN_IV_POINTS = 6
QUOTE_SIZE_VEV = 10
QUOTE_SIZE_DELTA = 8
EDGE_CLIP = 35.0
VEGA_SCALE_REF = 80.0

HYD_MM_WIDTH = 3
HYD_QUOTE_SIZE = 5

WING_STRIKES = {4000, 4500, 6500}
HS_WIDE = 8.0
EDGE_MULT_WIDE = 0.85

SHOCK_ABS_DS = 2.0
CORE_STRIKES = {5000, 5100, 5200, 5300}
CORE_SHOCK_SIZE_BOOST = 3


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
        out: dict[str, list[Order]] = {}

        du = depths.get(U)
        dh = depths.get(H)
        if du is None:
            return out, 0, json.dumps(store, separators=(",", ":"))

        s_mid = micro_mid(du)
        if s_mid is None:
            return out, 0, json.dumps(store, separators=(",", ":"))

        s_prev = store.get("prev_s_mid")
        shock = isinstance(s_prev, (int, float)) and math.isfinite(float(s_prev)) and abs(float(s_mid) - float(s_prev)) >= SHOCK_ABS_DS
        store["prev_s_mid"] = float(s_mid)

        h_mid = micro_mid(dh) if dh is not None else None
        csv_day = int(store.get("csv_day", -1))
        if csv_day < 0 or ts == 0:
            csv_day = infer_csv_day(float(s_mid), float(h_mid) if h_mid is not None else None)
            store["csv_day"] = csv_day

        T = t_years_effective(csv_day, ts)
        sqrtT = math.sqrt(T) if T > 0 else 1e-6

        # Collect node IVs by strike
        mids_v: dict[str, float] = {}
        iv_nodes: dict[str, float] = {}
        x_nodes: dict[str, float] = {}
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
            mids_v[v] = float(m)
            iv_nodes[v] = float(iv)
            x_nodes[v] = float(math.log(K / float(s_mid)) / sqrtT)

        if len(iv_nodes) < MIN_IV_POINTS:
            return out, 0, json.dumps(store, separators=(",", ":"))

        # Sticky EWMA at node level
        prev = store.get("node_iv")
        prev_map: dict[str, float] = prev if isinstance(prev, dict) else {}
        node_ema: dict[str, float] = {}
        a = NODE_EWMA_ALPHA
        for v, iv in iv_nodes.items():
            p = prev_map.get(v)
            if isinstance(p, (int, float)) and math.isfinite(float(p)):
                node_ema[v] = (1.0 - a) * float(p) + a * float(iv)
            else:
                node_ema[v] = float(iv)
        store["node_iv"] = node_ema

        # Wing regularization toward quadratic baseline before spline
        x_list = []
        y_list = []
        w_list = []
        for v, iv in node_ema.items():
            x = x_nodes[v]
            K = float(v.split("_")[1])
            vg = bs_vega(float(s_mid), K, T, max(iv, 1e-4), 0.0)
            x_list.append(x)
            y_list.append(iv)
            w_list.append(max(vg, 1e-6))
        x_arr = np.asarray(x_list, dtype=float)
        y_arr = np.asarray(y_list, dtype=float)
        w_arr = np.asarray(w_list, dtype=float)
        coeff_raw = np.polyfit(x_arr, y_arr, 2, w=w_arr)
        coef2 = coeff_raw
        y_base = np.polyval(coef2, x_arr)
        lam = WING_REG_LAMBDA
        y_reg = (1.0 - lam) * y_arr + lam * y_base

        # Sticky quadratic coefficients (same data as wing baseline; EMA in time like v2)
        ema = store.get("ema_coeff")
        if not isinstance(ema, list) or len(ema) != 3:
            ema = [0.15, 0.0, 0.24]
        pa = POLY_EWMA_ALPHA
        ema = [(1.0 - pa) * float(ema[i]) + pa * float(coeff_raw[i]) for i in range(3)]
        store["ema_coeff"] = ema
        c2, c1, c0 = float(ema[0]), float(ema[1]), float(ema[2])

        ord_idx = np.argsort(x_arr)
        x_sort = x_arr[ord_idx]
        y_sort = y_reg[ord_idx]
        span = float(x_sort[-1] - x_sort[0]) if len(x_sort) > 1 else 0.0
        if span > 0:
            xl = float(x_sort[0] - 0.25 * span)
            xr = float(x_sort[-1] + 0.25 * span)
            yl = float(np.polyval(coef2, xl))
            yr = float(np.polyval(coef2, xr))
            x_fit = np.concatenate([[xl], x_sort, [xr]])
            y_fit = np.concatenate([[yl], y_sort, [yr]])
        else:
            x_fit = x_sort
            y_fit = y_sort
        spline = CubicSpline(x_fit, y_fit, bc_type="natural")

        bw = float(FAIR_BLEND_POLY_W)
        if not (0.0 <= bw <= 1.0) or not math.isfinite(bw):
            bw = 0.35
        sw = 1.0 - bw

        # Quote loop
        for v in VOUCHERS:
            if v not in mids_v:
                continue
            dv = depths.get(v)
            if dv is None:
                continue
            K = int(v.split("_")[1])
            x = float(math.log(float(K) / float(s_mid)) / sqrtT)
            fair_iv_s = float(spline(x))
            fair_iv_s = max(1e-4, min(8.0, fair_iv_s))
            fair_iv_p = max(1e-4, min(8.0, float(np.polyval([c2, c1, c0], x))))
            price_s = bs_call_price(float(s_mid), float(K), T, fair_iv_s, 0.0)
            price_p = bs_call_price(float(s_mid), float(K), T, fair_iv_p, 0.0)
            fair = sw * price_s + bw * price_p
            mid = float(mids_v[v])
            fair_iv_g = max(1e-4, min(8.0, sw * fair_iv_s + bw * fair_iv_p))
            vega = bs_vega(float(s_mid), float(K), T, fair_iv_g, 0.0)
            edge = mid - fair
            wq = max(0.35, min(2.5, vega / VEGA_SCALE_REF))

            bb, ba = best_bid_ask(dv)
            if bb is None or ba is None:
                continue
            half_spread = 0.5 * float(ba - bb)
            need_edge = EDGE_MULT_WIDE * half_spread if (K in WING_STRIKES and half_spread >= HS_WIDE) else 0.0
            if need_edge > 0 and abs(edge) < need_edge:
                continue

            half_spread_floor = max(0.5, 0.3 * float(ba - bb))
            atm_w = math.exp(-0.5 * x * x)
            base_width = 1.0 + 1.4 * (1.0 - atm_w)
            shift = base_width + half_spread_floor + wq * max(-EDGE_CLIP, min(EDGE_CLIP, edge)) * 0.05

            pos_v = int(pos.get(v, 0))
            lim = LIMITS[v]

            bid_px = int(round(fair - shift))
            ask_px = int(round(fair + shift))
            bid_px = min(bid_px, ba - 1)
            ask_px = max(ask_px, bb + 1)
            if ask_px <= bid_px:
                ask_px = bid_px + 1

            q_buy = min(QUOTE_SIZE_VEV, lim - pos_v)
            q_sell = min(QUOTE_SIZE_VEV, lim + pos_v)
            if shock and K in CORE_STRIKES:
                q_buy = min(q_buy + CORE_SHOCK_SIZE_BOOST, lim - pos_v)
                q_sell = min(q_sell + CORE_SHOCK_SIZE_BOOST, lim + pos_v)
            if edge > 1.5:
                q_buy = min(q_buy + QUOTE_SIZE_DELTA, lim - pos_v)
            if edge < -1.5:
                q_sell = min(q_sell + QUOTE_SIZE_DELTA, lim + pos_v)

            ol: list[Order] = []
            if q_buy > 0 and bid_px > 0:
                ol.append(Order(v, bid_px, q_buy))
            if q_sell > 0:
                ol.append(Order(v, ask_px, -q_sell))
            if ol:
                out[v] = ol

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
