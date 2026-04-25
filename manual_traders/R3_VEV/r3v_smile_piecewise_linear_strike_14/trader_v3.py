"""
v3: PWL IV smile + IV scalping. Smooth underlying mid (EMA) for IV inversion and BS fair
(reduces micro-jitter vs raw S). Skip VEV_4000/4500; merge duplicate order prices; cap wing size.
"""
from __future__ import annotations

import json

from datamodel import Order, TradingState

from _r3v_smile_core import (
    book_walls,
    bs_call_price,
    bs_vega,
    infer_csv_day_from_smile,
    implied_vol_bisect,
    parse_td,
    pwl_iv_strike,
    synthetic_walls,
    t_years_effective,
)

STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
SKIP_TRADE = frozenset({4000, 4500})
WING = frozenset({6000, 6500})
U = "VELVETFRUIT_EXTRACT"
KNOTS = (5000, 5200, 5400)

LIMIT_VEV = 300
_TD_KEY = "r3v_pwl_smile"

THR_OPEN = 0.45
THR_CLOSE = 0.05
LOW_VEGA_THR_ADJ = 0.45
LOW_VEGA_CUTOFF = 1.2
THEO_NORM_WINDOW = 25
IV_SCALPING_WINDOW = 100
IV_SCALPING_THR = 0.55
WARMUP_STEPS = 12
S_EMA_WINDOW = 140
WING_CAP = 40


def _median(xs: list[float]) -> float | None:
    import math

    ys = [x for x in xs if x is not None and math.isfinite(x)]
    if not ys:
        return None
    ys.sort()
    return ys[len(ys) // 2]


def _ema(store: dict[str, float], key: str, window: int, value: float) -> float:
    old = float(store.get(key, 0.0))
    alpha = 2.0 / (window + 1.0)
    new = alpha * value + (1.0 - alpha) * old
    store[key] = new
    return new


def _knot_ivs_from_surface(strike_iv: dict[int, float | None]) -> tuple[float, float, float] | None:
    import math

    def ivs(ks: tuple[int, ...]) -> list[float]:
        out: list[float] = []
        for k in ks:
            v = strike_iv.get(k)
            if v is not None and math.isfinite(v) and v > 0:
                out.append(float(v))
        return out

    m0 = _median(ivs((5000, 5100)))
    m1 = _median(ivs((5100, 5200, 5300)))
    m2 = _median(ivs((5300, 5400, 5500)))
    if m0 is None or m1 is None or m2 is None:
        return None

    def clip(x: float) -> float:
        return max(0.04, min(3.5, x))

    return (clip(m0), clip(m1), clip(m2))


def _scalp_orders(
    wall_mid: float,
    best_bid: int,
    best_ask: int,
    pos: int,
    lim: int,
    cur_diff: float,
    mean_diff: float,
    switch_mean: float,
    vega: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    bids: list[tuple[int, int]] = []
    asks: list[tuple[int, int]] = []
    max_buy = lim - pos
    max_sell = lim + pos
    low_adj = LOW_VEGA_THR_ADJ if vega <= LOW_VEGA_CUTOFF else 0.0
    if switch_mean >= IV_SCALPING_THR:
        if cur_diff - wall_mid + best_bid - mean_diff >= (THR_OPEN + low_adj) and max_sell > 0:
            asks.append((best_bid, max_sell))
        if cur_diff - wall_mid + best_bid - mean_diff >= THR_CLOSE and pos > 0:
            asks.append((best_bid, pos))
        elif cur_diff - wall_mid + best_ask - mean_diff <= -(THR_OPEN + low_adj) and max_buy > 0:
            bids.append((best_ask, max_buy))
        if cur_diff - wall_mid + best_ask - mean_diff <= -THR_CLOSE and pos < 0:
            bids.append((best_ask, -pos))
    else:
        if pos > 0:
            asks.append((best_bid, pos))
        elif pos < 0:
            bids.append((best_ask, -pos))
    return bids, asks


def _merge_levels(levels: list[tuple[int, int]]) -> dict[int, int]:
    m: dict[int, int] = {}
    for px, q in levels:
        m[int(px)] = m.get(int(px), 0) + int(q)
    return m


class Trader:
    def run(self, state: TradingState):
        import math

        store = parse_td(getattr(state, "traderData", None))
        ema: dict[str, float] = store.get(_TD_KEY) if isinstance(store.get(_TD_KEY), dict) else {}
        if not isinstance(ema, dict):
            ema = {}
        ema = {str(k): float(v) for k, v in ema.items() if isinstance(v, (int, float))}

        depths = getattr(state, "order_depths", None) or {}
        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < WARMUP_STEPS:
            store[_TD_KEY] = ema
            return {}, 0, json.dumps(store, separators=(",", ":"))

        if U not in depths:
            store[_TD_KEY] = ema
            return {}, 0, json.dumps(store, separators=(",", ":"))

        _bw, _aw, ubb, uba, _ = book_walls(depths[U])
        if ubb is None or uba is None:
            store[_TD_KEY] = ema
            return {}, 0, json.dumps(store, separators=(",", ":"))
        S = 0.5 * (ubb + uba)
        if "S_ema" not in ema or ema["S_ema"] <= 0:
            ema["S_ema"] = float(S)
        else:
            _ema(ema, "S_ema", S_EMA_WINDOW, float(S))
        S_fair = float(ema["S_ema"])

        strike_mids: dict[int, float] = {}
        for k, sym in zip(STRIKES, VOUCHERS):
            d = depths.get(sym)
            if d is None:
                continue
            bw0, aw0, bb, ba, _wm0 = book_walls(d)
            _bw, _aw, wm2, bb2, ba2 = synthetic_walls(bw0, aw0, bb, ba)
            mid = wm2
            if mid is None and bb2 is not None and ba2 is not None:
                mid = 0.5 * (bb2 + ba2)
            if mid is not None and math.isfinite(mid):
                strike_mids[k] = float(mid)

        if "csv_day_est" not in ema and len(strike_mids) >= 6:
            ema["csv_day_est"] = float(infer_csv_day_from_smile(S_fair, strike_mids, ts, STRIKES))
        d_csv = int(ema.get("csv_day_est", 0))
        T = t_years_effective(d_csv, ts)

        strike_iv: dict[int, float | None] = {}
        for K in STRIKES:
            mid = strike_mids.get(K)
            if mid is None:
                strike_iv[K] = None
                continue
            strike_iv[K] = implied_vol_bisect(mid, S_fair, float(K), T)

        knots_iv = _knot_ivs_from_surface(strike_iv)
        if knots_iv is None:
            store[_TD_KEY] = ema
            return {}, 0, json.dumps(store, separators=(",", ":"))

        orders: dict[str, list[Order]] = {}
        for k, sym in zip(STRIKES, VOUCHERS):
            if k in SKIP_TRADE:
                continue
            d = depths.get(sym)
            if d is None:
                continue
            bw0, aw0, bb, ba, _wm0 = book_walls(d)
            _bw, _aw, wm2, bb2, ba2 = synthetic_walls(bw0, aw0, bb, ba)
            if wm2 is None or bb2 is None or ba2 is None:
                continue
            sig = pwl_iv_strike(float(k), KNOTS, knots_iv)
            if not math.isfinite(sig) or sig <= 0:
                continue
            theo = bs_call_price(S_fair, float(k), T, sig)
            if not math.isfinite(theo):
                continue
            cur_diff = float(wm2) - theo
            mean_diff = _ema(ema, f"{sym}_theo_diff", THEO_NORM_WINDOW, cur_diff)
            switch_mean = _ema(ema, f"{sym}_avg_devs", IV_SCALPING_WINDOW, abs(cur_diff - mean_diff))
            vega = bs_vega(S_fair, float(k), T, sig)
            pos = int((getattr(state, "position", None) or {}).get(sym, 0))
            bids, asks = _scalp_orders(
                float(wm2), int(bb2), int(ba2), pos, LIMIT_VEV, cur_diff, mean_diff, switch_mean, vega
            )
            cap = WING_CAP if k in WING and vega < 0.55 else LIMIT_VEV
            ol: list[Order] = []
            for px, q in _merge_levels(asks).items():
                q = min(int(q), cap, LIMIT_VEV + pos)
                if q > 0:
                    ol.append(Order(sym, int(px), -q))
            for px, q in _merge_levels(bids).items():
                q = min(int(q), cap, LIMIT_VEV - pos)
                if q > 0:
                    ol.append(Order(sym, int(px), q))
            if ol:
                orders[sym] = ol

        store[_TD_KEY] = ema
        return orders, 0, json.dumps(store, separators=(",", ":"))
