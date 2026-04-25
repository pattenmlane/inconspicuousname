"""
Round 3 VEV: local smile (closest strikes) + quadratic IV in log-moneyness; MM from fitted surface.

Iteration 1 vs v0: fit smile on closest **3** strikes only (analysis showed k=6 had lower IV RMSE on tape).

TTE/DTE: round3description.txt example + round3work/plotting/.../plot_iv_smile_round3.py —
CSV historical day d has calendar DTE at open = 8 - d; intraday DTE_eff = DTE_open - (ts//100)/10000;
T = DTE_eff/365, r=0. Backtester does not expose CSV day in state; we infer session day at t=0 from
opening VELVETFRUIT_EXTRACT mid (Round 3 tape anchors: 5250.0->day0, 5245.0->day1, 5267.5->day2).

Params: CLOSEST_K=3, MM_EDGE=2, REFIT_EVERY=80.
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

CLOSEST_K = 3
MM_EDGE = 2
HYDRO_EDGE = 2
ORDER_SIZE_VEV = 18
ORDER_SIZE_U = 12
ORDER_SIZE_H = 15
SKEW_PER_UNIT = 0.08
EMA_ALPHA = 0.12
REFIT_EVERY = 80


def _best_bid_ask(d: OrderDepth) -> tuple[int, int] | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    bb = max(d.buy_orders)
    ba = min(d.sell_orders)
    return bb, ba


def _mid(d: OrderDepth) -> float | None:
    ba = _best_bid_ask(d)
    if ba is None:
        return None
    return (ba[0] + ba[1]) / 2.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 1e-12:
        return max(S - K, 0.0)
    if sigma <= 1e-12:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def implied_vol_bisect(price: float, S: float, K: float, T: float, r: float = 0.0) -> float | None:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6 or price >= S - 1e-6 or S <= 0 or K <= 0 or T <= 1e-12:
        return None
    lo, hi = 1e-4, 12.0
    flo = bs_call_price(S, K, T, lo, r) - price
    fhi = bs_call_price(S, K, T, hi, r) - price
    if flo > 0 or fhi < 0:
        return None
    for _ in range(34):
        mid = 0.5 * (lo + hi)
        if bs_call_price(S, K, T, mid, r) >= price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def dte_open_from_csv_day(csv_day: int) -> float:
    return float(8 - int(csv_day))


def dte_effective(csv_day: int, timestamp: int) -> float:
    prog = (int(timestamp) // 100) / 10_000.0
    return max(dte_open_from_csv_day(csv_day) - prog, 1e-6)


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


def iv_surface_value(S: float, K: float, coeffs: tuple[float, float, float] | None) -> float | None:
    if coeffs is None or S <= 0 or K <= 0:
        return None
    x = math.log(K / S)
    a, b, c = coeffs
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
            du = depths.get(U)
            if du:
                m = _mid(du)
                if m is not None:
                    key = round(m * 2) / 2.0
                    if key in ANCHOR_S0_TO_CSV_DAY:
                        td["csv_day"] = ANCHOR_S0_TO_CSV_DAY[key]
                    else:
                        nearest = min(ANCHOR_S0_TO_CSV_DAY, key=lambda s: abs(s - m))
                        if abs(nearest - m) < 2.0:
                            td["csv_day"] = ANCHOR_S0_TO_CSV_DAY[nearest]
        csv_day = int(td.get("csv_day", 0))

        du = depths.get(U)
        if not du:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        m_u = _mid(du)
        if m_u is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        ema = float(td.get("ema_S", m_u))
        ema += EMA_ALPHA * (m_u - ema)
        td["ema_S"] = ema
        S = ema

        T = dte_effective(csv_day, ts) / 365.0

        coeffs: tuple[float, float, float] | None = None
        prev = td.get("smile_coeffs")
        last_ref = int(td.get("smile_ts", -999999))
        if (
            isinstance(prev, list)
            and len(prev) == 3
            and ts - last_ref < REFIT_EVERY * 100
        ):
            coeffs = (float(prev[0]), float(prev[1]), float(prev[2]))
        else:
            pairs: list[tuple[float, float, float]] = []
            for K in STRIKES:
                sym = VEV_BY_K[K]
                d = depths.get(sym)
                if not d:
                    continue
                m = _mid(d)
                if m is None:
                    continue
                iv = implied_vol_bisect(m, S, float(K), T, 0.0)
                if iv is None:
                    continue
                pairs.append((float(K), abs(float(K) - S), iv))

            pairs.sort(key=lambda t: t[1])
            take = pairs[: min(CLOSEST_K, len(pairs))]
            if len(take) < 2:
                coeffs = None
            else:
                xs = [math.log(k / S) for k, _, iv in take]
                ys = [iv for _, _, iv in take]
                deg = 2 if len(take) >= 3 else 1
                if deg == 1:
                    n = len(xs)
                    sx, sy = sum(xs), sum(ys)
                    sxx = sum(x * x for x in xs)
                    sxy = sum(x * y for x, y in zip(xs, ys))
                    det = n * sxx - sx * sx
                    if abs(det) < 1e-12:
                        coeffs = (0.0, 0.0, sum(ys) / len(ys))
                    else:
                        b = (n * sxy - sx * sy) / det
                        c = (sxx * sy - sx * sxy) / det
                        coeffs = (0.0, b, c)
                else:
                    c2 = polyfit2(xs, ys)
                    coeffs = c2 if c2 else (0.0, 0.0, sum(ys) / len(ys))
            if coeffs is not None:
                td["smile_coeffs"] = [coeffs[0], coeffs[1], coeffs[2]]
                td["smile_ts"] = ts

        orders: dict[str, list[Order]] = {}

        def mm_sym(sym: str, d: OrderDepth, fair: float, edge: int, sz: int) -> list[Order]:
            ba = _best_bid_ask(d)
            if ba is None:
                return []
            bb, ba_p = ba
            p = int(pos.get(sym, 0))
            skew = int(round(SKEW_PER_UNIT * p))
            bid_p = min(bb + 1, int(math.floor(fair - edge - skew)))
            ask_p = max(ba_p - 1, int(math.ceil(fair + edge - skew)))
            lim = LIMITS[sym]
            out: list[Order] = []
            if p < lim and bid_p > 0:
                out.append(Order(sym, bid_p, min(sz, lim - p)))
            if p > -lim and ask_p > 0:
                out.append(Order(sym, ask_p, -min(sz, lim + p)))
            return out

        if H in depths:
            dh = depths[H]
            mh = _mid(dh)
            if mh is not None:
                orders[H] = mm_sym(H, dh, mh, HYDRO_EDGE, ORDER_SIZE_H)

        orders[U] = mm_sym(U, du, S, MM_EDGE, ORDER_SIZE_U)

        for K in STRIKES:
            sym = VEV_BY_K[K]
            d = depths.get(sym)
            if not d or coeffs is None:
                continue
            iv_hat = iv_surface_value(S, float(K), coeffs)
            if iv_hat is None:
                continue
            fair = bs_call_price(S, float(K), T, iv_hat, 0.0)
            orders[sym] = mm_sym(sym, d, fair, MM_EDGE, ORDER_SIZE_VEV)

        return orders, 0, json.dumps(td, separators=(",", ":"))
