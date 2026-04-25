"""
Smile MM iteration v2: same as v1 (EMA-smoothed quadratic IV smile) but **wider extract**
half-spread on historical CSV day 2 only (TTE 6d at open — shortest slice in our tapes),
where aggressive underlying MM lost the most in v0/v1. Day index from TestRunner stack
when available (backtest); else no bump.
"""
from __future__ import annotations

import inspect
import json
import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
VEV_PRODUCTS = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
LIMITS = {**{p: 300 for p in VEV_PRODUCTS}, UNDERLYING: 200, HYDRO: 200}

RISK_FREE = 0.0
BASE_HALF_SPREAD = 1.2
WING_KM_SQ_SCALE = 420.0
SIZE_VEV = 22
SIZE_EXTRACT = 10
SIZE_HYDRO = 10
SKEW_PER_UNIT = 0.045
EXTRACT_HEDGE_BETA = 0.35
EXTRACT_HALF_DEFAULT = 2.55
EXTRACT_HALF_SHORT_TTE = 3.85
HYDRO_HALF = 2.0
IV_FIT_WEIGHT_MIN = 0.15
SMILE_EMA_ALPHA = 0.08
_STORE_KEY_SMOOTH = "smile_abc_smooth"


def _csv_day_from_backtest_stack() -> int | None:
    for fr in inspect.stack():
        data = fr.frame.f_locals.get("data")
        if data is not None and hasattr(data, "day_num"):
            try:
                return int(getattr(data, "day_num"))
            except (TypeError, ValueError):
                continue
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def dte_from_csv_day(day: int) -> int:
    return 8 - int(day)


def intraday_progress(timestamp: int) -> float:
    return (int(timestamp) // 100) / 10_000.0


def dte_effective(day: int, timestamp: int) -> float:
    return max(float(dte_from_csv_day(day)) - intraday_progress(timestamp), 1e-6)


def t_years_effective(day: int, timestamp: int) -> float:
    return dte_effective(day, timestamp) / 365.0


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0.0 or sigma <= 1e-12:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    return norm_cdf(d1)


def implied_vol_newton(mid: float, S: float, K: float, T: float, r: float = 0.0) -> float | None:
    if mid <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-6:
        return None
    sig = 0.25
    for _ in range(40):
        th = bs_call_price(S, K, T, sig, r)
        vega_denom = S * math.sqrt(T) * math.exp(-0.5 * (((math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))) ** 2)) / math.sqrt(2 * math.pi)
        if vega_denom < 1e-12:
            break
        diff = th - mid
        if abs(diff) < 1e-4:
            return max(sig, 1e-6)
        sig -= diff / vega_denom
        sig = min(max(sig, 1e-4), 5.0)
    return None


def strike_from_product(p: str) -> float:
    return float(p.split("_", 1)[1])


def book_mid(depth: OrderDepth | None) -> tuple[float, float, float] | None:
    if depth is None:
        return None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if ba <= bb:
        return None
    return float(bb), float(ba), 0.5 * (bb + ba)


def polyfit_quadratic(
    xs: list[float], ys: list[float], weights: list[float]
) -> tuple[float, float, float] | None:
    if len(xs) < 4:
        return None
    n = len(xs)
    s_w = sum(weights)
    if s_w <= 0:
        return None
    sx = sum(weights[i] * xs[i] for i in range(n))
    sx2 = sum(weights[i] * xs[i] ** 2 for i in range(n))
    sx3 = sum(weights[i] * xs[i] ** 3 for i in range(n))
    sx4 = sum(weights[i] * xs[i] ** 4 for i in range(n))
    sy = sum(weights[i] * ys[i] for i in range(n))
    sxy = sum(weights[i] * xs[i] * ys[i] for i in range(n))
    sx2y = sum(weights[i] * xs[i] * xs[i] * ys[i] for i in range(n))
    S = [[s_w, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]]
    R = [sy, sxy, sx2y]

    def det3(m: list[list[float]]) -> float:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    d0 = det3(S)
    if abs(d0) < 1e-18:
        return None
    S0 = [[R[0], S[0][1], S[0][2]], [R[1], S[1][1], S[1][2]], [R[2], S[2][1], S[2][2]]]
    S1 = [[S[0][0], R[0], S[0][2]], [S[1][0], R[1], S[1][2]], [S[2][0], R[2], S[2][2]]]
    S2 = [[S[0][0], S[0][1], R[0]], [S[1][0], S[1][1], R[1]], [S[2][0], S[2][1], R[2]]]
    c = det3(S0) / d0
    b = det3(S1) / d0
    a = det3(S2) / d0
    return c, b, a


def _ema_triplet(
    prev: tuple[float, float, float] | None,
    nxt: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    if prev is None:
        return nxt
    return tuple(prev[i] * (1.0 - alpha) + nxt[i] * alpha for i in range(3))


class Trader:
    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths = getattr(state, "order_depths", None) or {}
        positions = getattr(state, "position", None) or {}

        csv_day = _csv_day_from_backtest_stack()
        if csv_day is None:
            csv_day = int(store.get("csv_day_hint", 0))
        store["csv_day_hint"] = csv_day

        extract_half = EXTRACT_HALF_SHORT_TTE if csv_day >= 2 else EXTRACT_HALF_DEFAULT

        prev_smooth = store.get(_STORE_KEY_SMOOTH)
        if isinstance(prev_smooth, list) and len(prev_smooth) == 3:
            prev_t = (float(prev_smooth[0]), float(prev_smooth[1]), float(prev_smooth[2]))
        else:
            prev_t = None

        if UNDERLYING not in depths:
            if prev_t is not None:
                store[_STORE_KEY_SMOOTH] = list(prev_t)
            return {}, 0, json.dumps(store, separators=(",", ":"))

        depth_u = depths.get(UNDERLYING)
        bu = book_mid(depth_u)
        if bu is None:
            if prev_t is not None:
                store[_STORE_KEY_SMOOTH] = list(prev_t)
            return {}, 0, json.dumps(store, separators=(",", ":"))
        _, _, mid_u = bu
        T = t_years_effective(csv_day, int(getattr(state, "timestamp", 0)))

        vev_products_ok: list[str] = []
        iv_points: list[tuple[str, float, float, float]] = []
        for p in VEV_PRODUCTS:
            if p not in depths:
                continue
            vev_products_ok.append(p)
            d = depths.get(p)
            b = book_mid(d)
            if b is None:
                continue
            _, _, mid = b
            K = strike_from_product(p)
            iv = implied_vol_newton(mid, mid_u, K, T, RISK_FREE)
            if iv is None:
                continue
            km = math.log(K / mid_u)
            spr = b[1] - b[0]
            w = 1.0 / max(spr, IV_FIT_WEIGHT_MIN)
            iv_points.append((p, km, iv, w))

        coeffs_raw = polyfit_quadratic(
            [t[1] for t in iv_points], [t[2] for t in iv_points], [t[3] for t in iv_points]
        )
        if coeffs_raw is not None:
            coeffs_t = _ema_triplet(prev_t, coeffs_raw, SMILE_EMA_ALPHA)
            store[_STORE_KEY_SMOOTH] = list(coeffs_t)
        elif prev_t is not None:
            coeffs_t = prev_t
        else:
            coeffs_t = None

        orders: dict[str, list[Order]] = {}

        def fair_iv_at_km(km: float) -> float:
            if coeffs_t is None:
                return 0.22
            c, b, a = coeffs_t
            return max(c + b * km + a * km * km, 0.03)

        net_call_delta_units = 0.0
        for p in vev_products_ok:
            d = depths.get(p)
            b = book_mid(d)
            if b is None:
                continue
            _, _, mid = b
            K = strike_from_product(p)
            km = math.log(K / mid_u)
            sig = fair_iv_at_km(km)
            fair = bs_call_price(mid_u, K, T, sig, RISK_FREE)
            delt = bs_call_delta(mid_u, K, T, sig, RISK_FREE)
            pos = int(positions.get(p, 0))
            lim = LIMITS[p]
            skew = SKEW_PER_UNIT * (pos / max(lim, 1))
            fair_adj = fair - skew * (b[1] - b[0])

            wing_w = WING_KM_SQ_SCALE * (km ** 2)
            half = BASE_HALF_SPREAD + wing_w
            bid_p = int(round(fair_adj - half))
            ask_p = int(round(fair_adj + half))
            bid_p = min(bid_p, int(b[1]) - 1)
            ask_p = max(ask_p, int(b[0]) + 1)
            if bid_p >= ask_p:
                continue

            qb = min(SIZE_VEV, lim - pos)
            qs = min(SIZE_VEV, lim + pos)
            ol: list[Order] = []
            if qb > 0 and bid_p > 0:
                ol.append(Order(p, bid_p, qb))
            if qs > 0 and ask_p > 0:
                ol.append(Order(p, ask_p, -qs))
            if ol:
                orders[p] = ol

            net_call_delta_units += delt * float(pos)

        pos_u = int(positions.get(UNDERLYING, 0))
        lim_u = LIMITS[UNDERLYING]
        bu2 = book_mid(depths.get(UNDERLYING))
        if bu2 is not None:
            hedge_shift = EXTRACT_HEDGE_BETA * net_call_delta_units
            fair_x = bu2[2] - hedge_shift
            bid_x = int(round(fair_x - extract_half))
            ask_x = int(round(fair_x + extract_half))
            bid_x = min(bid_x, int(bu2[1]) - 1)
            ask_x = max(ask_x, int(bu2[0]) + 1)
            if bid_x < ask_x:
                qb = min(SIZE_EXTRACT, lim_u - pos_u)
                qs = min(SIZE_EXTRACT, lim_u + pos_u)
                olu: list[Order] = []
                if qb > 0:
                    olu.append(Order(UNDERLYING, bid_x, qb))
                if qs > 0:
                    olu.append(Order(UNDERLYING, ask_x, -qs))
                if olu:
                    orders[UNDERLYING] = olu

        if HYDRO in depths:
            bh = book_mid(depths[HYDRO])
            if bh is not None:
                pos_h = int(positions.get(HYDRO, 0))
                lim_h = LIMITS[HYDRO]
                fair_h = bh[2]
                bid_h = int(round(fair_h - HYDRO_HALF))
                ask_h = int(round(fair_h + HYDRO_HALF))
                bid_h = min(bid_h, int(bh[1]) - 1)
                ask_h = max(ask_h, int(bh[0]) + 1)
                if bid_h < ask_h:
                    qb = min(SIZE_HYDRO, lim_h - pos_h)
                    qs = min(SIZE_HYDRO, lim_h + pos_h)
                    olh: list[Order] = []
                    if qb > 0:
                        olh.append(Order(HYDRO, bid_h, qb))
                    if qs > 0:
                        olh.append(Order(HYDRO, ask_h, -qs))
                    if olh:
                        orders[HYDRO] = olh

        return orders, 0, json.dumps(store, separators=(",", ":"))
