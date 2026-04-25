"""
Round 3 VEV: local smile (closest strikes) + quadratic IV in log-moneyness; MM from fitted surface.

Iteration 2 (same thesis): keep local smile on closest-6, but make voucher quote widths and sizes
vega-aware so high-vega near-ATM strikes quote tighter/larger than wings.

TTE/DTE: round3description.txt + round3work plotting convention:
CSV day d -> DTE_open = 8-d, intraday dte_eff = DTE_open - (ts//100)/10000, T = dte_eff/365.
"""
from __future__ import annotations

import json
import math
from datamodel import Order, OrderDepth, TradingState
from typing import Any

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VEV_BY_K = {k: f"VEV_{k}" for k in STRIKES}
ALL_VEV = list(VEV_BY_K.values())
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"

LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in ALL_VEV},
}

ANCHOR_S0_TO_CSV_DAY = {
    5250.0: 0,
    5245.0: 1,
    5267.5: 2,
}

CLOSEST_K = 6
MM_EDGE_U = 2
MM_EDGE_H = 2
BASE_EDGE_VEV = 3
ORDER_SIZE_U = 12
ORDER_SIZE_H = 15
BASE_ORDER_SIZE_VEV = 12
SKEW_PER_UNIT = 0.08
EMA_ALPHA = 0.12
REFIT_EVERY = 80


def _best_bid_ask(d: OrderDepth) -> tuple[int, int] | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return max(d.buy_orders), min(d.sell_orders)


def _mid(d: OrderDepth) -> float | None:
    ba = _best_bid_ask(d)
    if ba is None:
        return None
    return (ba[0] + ba[1]) / 2.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(s: float, k: float, t: float, sigma: float) -> float:
    if t <= 1e-12 or sigma <= 1e-12:
        return max(s - k, 0.0)
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / v
    d2 = d1 - v
    return s * _norm_cdf(d1) - k * _norm_cdf(d2)


def bs_vega(s: float, k: float, t: float, sigma: float) -> float:
    if t <= 1e-12 or sigma <= 1e-12:
        return 0.0
    v = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / v
    return s * _norm_pdf(d1) * math.sqrt(t)


def implied_vol_bisect(price: float, s: float, k: float, t: float) -> float | None:
    intrinsic = max(s - k, 0.0)
    if price <= intrinsic + 1e-6 or price >= s - 1e-6 or s <= 0 or k <= 0 or t <= 1e-12:
        return None
    lo, hi = 1e-4, 12.0
    flo = bs_call_price(s, k, t, lo) - price
    fhi = bs_call_price(s, k, t, hi) - price
    if flo > 0 or fhi < 0:
        return None
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        if bs_call_price(s, k, t, mid) >= price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def dte_effective(csv_day: int, timestamp: int) -> float:
    d_open = 8.0 - float(csv_day)
    intra = (int(timestamp) // 100) / 10_000.0
    return max(d_open - intra, 1e-6)


def polyfit2(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    n = len(xs)
    if n < 2:
        return None
    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x * x * x for x in xs)
    s4 = sum(x * x * x * x for x in xs)
    t0 = sum(ys)
    t1 = sum(x * y for x, y in zip(xs, ys))
    t2 = sum(x * x * y for x, y in zip(xs, ys))
    if n == 2:
        det = s0 * s2 - s1 * s1
        if abs(det) < 1e-12:
            c = (t0 + t1) / (s0 + s1) if abs(s0 + s1) > 1e-12 else t0 / s0
            return 0.0, 0.0, c
        c = (s2 * t0 - s1 * t1) / det
        b = (-s1 * t0 + s0 * t1) / det
        return 0.0, b, c
    det = (
        s0 * (s2 * s4 - s3 * s3)
        - s1 * (s1 * s4 - s2 * s3)
        + s2 * (s1 * s3 - s2 * s2)
    )
    if abs(det) < 1e-18:
        return polyfit2(xs[:2], ys[:2])
    c = (
        t0 * (s2 * s4 - s3 * s3)
        - t1 * (s1 * s4 - s2 * s3)
        + t2 * (s1 * s3 - s2 * s2)
    ) / det
    b = (
        s0 * (t1 * s4 - t2 * s3)
        - s1 * (t0 * s4 - t2 * s2)
        + s2 * (t0 * s3 - t1 * s2)
    ) / det
    a = (
        s0 * (s2 * t2 - s3 * t1)
        - s1 * (s1 * t2 - s3 * t0)
        + s2 * (s1 * t1 - s2 * t0)
    ) / det
    return a, b, c


def iv_surface_value(s: float, k: float, coeffs: tuple[float, float, float] | None) -> float | None:
    if coeffs is None or s <= 0 or k <= 0:
        return None
    a, b, c = coeffs
    x = math.log(k / s)
    sig = a * x * x + b * x + c
    if sig < 1e-4 or sig > 10.0:
        return None
    return sig


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict[str, Any] = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        depths = state.order_depths
        pos = state.position
        ts = int(state.timestamp)

        if ts == 0:
            d_u = depths.get(U)
            if d_u:
                m_u = _mid(d_u)
                if m_u is not None:
                    key = round(m_u * 2) / 2.0
                    if key in ANCHOR_S0_TO_CSV_DAY:
                        td["csv_day"] = ANCHOR_S0_TO_CSV_DAY[key]
                    else:
                        nearest = min(ANCHOR_S0_TO_CSV_DAY, key=lambda x: abs(x - m_u))
                        if abs(nearest - m_u) < 2.0:
                            td["csv_day"] = ANCHOR_S0_TO_CSV_DAY[nearest]
        csv_day = int(td.get("csv_day", 0))

        d_u = depths.get(U)
        if not d_u:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        m_u = _mid(d_u)
        if m_u is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        ema = float(td.get("ema_S", m_u))
        ema += EMA_ALPHA * (m_u - ema)
        td["ema_S"] = ema
        s = ema
        t = dte_effective(csv_day, ts) / 365.0

        coeffs: tuple[float, float, float] | None = None
        prev = td.get("smile_coeffs")
        last_ref = int(td.get("smile_ts", -999999))
        if isinstance(prev, list) and len(prev) == 3 and ts - last_ref < REFIT_EVERY * 100:
            coeffs = (float(prev[0]), float(prev[1]), float(prev[2]))
        else:
            pairs: list[tuple[float, float, float]] = []
            for k in STRIKES:
                sym = VEV_BY_K[k]
                d = depths.get(sym)
                if not d:
                    continue
                m = _mid(d)
                if m is None:
                    continue
                iv = implied_vol_bisect(m, s, float(k), t)
                if iv is None:
                    continue
                pairs.append((float(k), abs(float(k) - s), iv))
            pairs.sort(key=lambda x: x[1])
            use = pairs[: min(CLOSEST_K, len(pairs))]
            if len(use) >= 2:
                xs = [math.log(k / s) for k, _, _ in use]
                ys = [iv for _, _, iv in use]
                coeffs = polyfit2(xs, ys) if len(use) >= 3 else (0.0, 0.0, sum(ys) / len(ys))
            if coeffs is not None:
                td["smile_coeffs"] = [coeffs[0], coeffs[1], coeffs[2]]
                td["smile_ts"] = ts

        orders: dict[str, list[Order]] = {}

        def mm_sym(sym: str, d: OrderDepth, fair: float, edge: int, size: int) -> list[Order]:
            ba = _best_bid_ask(d)
            if ba is None:
                return []
            bb, ap = ba
            p = int(pos.get(sym, 0))
            skew = int(round(SKEW_PER_UNIT * p))
            bid = min(bb + 1, int(math.floor(fair - edge - skew)))
            ask = max(ap - 1, int(math.ceil(fair + edge - skew)))
            lim = LIMITS[sym]
            out: list[Order] = []
            if p < lim and bid > 0:
                out.append(Order(sym, bid, min(size, lim - p)))
            if p > -lim and ask > 0:
                out.append(Order(sym, ask, -min(size, lim + p)))
            return out

        if H in depths:
            d_h = depths[H]
            m_h = _mid(d_h)
            if m_h is not None:
                orders[H] = mm_sym(H, d_h, m_h, MM_EDGE_H, ORDER_SIZE_H)

        orders[U] = mm_sym(U, d_u, s, MM_EDGE_U, ORDER_SIZE_U)

        for k in STRIKES:
            sym = VEV_BY_K[k]
            d = depths.get(sym)
            if not d or coeffs is None:
                continue
            sig = iv_surface_value(s, float(k), coeffs)
            if sig is None:
                continue
            fair = bs_call_price(s, float(k), t, sig)
            vega = bs_vega(s, float(k), t, sig)

            # Vega-aware microstructure: tighter + larger near ATM, wider + smaller on wings.
            edge_adj = 0
            if vega > 220:
                edge_adj = -1
            elif vega < 40:
                edge_adj = 1
            edge = max(1, BASE_EDGE_VEV + edge_adj)

            size = BASE_ORDER_SIZE_VEV
            if vega > 220:
                size = 20
            elif vega > 120:
                size = 16
            elif vega < 20:
                size = 8

            orders[sym] = mm_sym(sym, d, fair, edge, size)

        return orders, 0, json.dumps(td, separators=(",", ":"))
