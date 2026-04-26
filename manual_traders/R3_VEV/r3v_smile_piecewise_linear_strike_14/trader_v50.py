"""
v48 + joint tight book gate (round3work/vouchers_final_strategy/STRATEGY.txt):
when VEV_5200 and VEV_5300 both have L1 spread <=2 (ask1-bid1), scale down THR_OPEN/THR_CLOSE
(easier to trade the PWL surface); when either is wide, scale up (stricter) — see
r3_tight_spread_summary.txt (forward extract mid more favorable on average when both tight).
"""
from __future__ import annotations

import json

from datamodel import Order, TradingState

from _r3v_smile_core import (
    book_walls,
    bs_call_price,
    bs_gamma,
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
SYM_5200 = "VEV_5200"
SYM_5300 = "VEV_5300"

LIMIT_VEV = 300
_TD_KEY = "r3v_pwl_smile"

THR_OPEN = 0.48
THR_CLOSE = 0.06
LOW_VEGA_THR_ADJ = 0.5
LOW_VEGA_CUTOFF = 1.1
THEO_NORM_WINDOW = 25
IV_SCALPING_WINDOW = 110
IV_SCALPING_THR = 0.58
WARMUP_STEPS = 15
S_EMA_WINDOW = 140
WING_CAP = 40
STRIKE_5000_CAP = 1
MIN_SPREAD = 5
STRIKE_SPREAD_FLOOR = {5000: 5, 5100: 4, 5200: 3, 5300: 2, 5400: 2, 5500: 2}

# Shared thesis: both legs <= TH at same time (STRATEGY.txt, TH=2)
TIGHT_5200_5300_MAX = 2
# Easier PWL scalps when surface is tradable; stricter when either book is wide
JOINT_TIGHT_THR_MULT = 0.92
WIDE_REGIME_THR_MULT = 1.04

EXTRACT_SHOCK_ABS = 4.0
SHOCK_THR_MULT = 0.14
GAMMA_SCALE = 8000.0


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


def _l1_spread_ticks(depth) -> int | None:
    """Top-of-book spread in same units as tape (ask1 - bid1)."""
    import math

    if depth is None:
        return None
    bw0, aw0, bb, ba, _wm0 = book_walls(depth)
    _bw, _aw, _wm, bb2, ba2 = synthetic_walls(bw0, aw0, bb, ba)
    if bb2 is None or ba2 is None:
        return None
    w = 0.5 * (bb2 + ba2) if _wm is None else _wm
    if not math.isfinite(float(w)):
        return None
    return int(ba2) - int(bb2)


def _joint_tight_gate(depths) -> bool:
    d52 = depths.get(SYM_5200)
    d53 = depths.get(SYM_5300)
    if d52 is None or d53 is None:
        return False
    s52 = _l1_spread_ticks(d52)
    s53 = _l1_spread_ticks(d53)
    if s52 is None or s53 is None:
        return False
    return s52 <= TIGHT_5200_5300_MAX and s53 <= TIGHT_5200_5300_MAX


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
    thr_open_e: float,
    thr_close_e: float,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    bids: list[tuple[int, int]] = []
    asks: list[tuple[int, int]] = []
    max_buy = lim - pos
    max_sell = lim + pos
    low_adj = LOW_VEGA_THR_ADJ if vega <= LOW_VEGA_CUTOFF else 0.0
    if switch_mean >= IV_SCALPING_THR:
        if cur_diff - wall_mid + best_bid - mean_diff >= (thr_open_e + low_adj) and max_sell > 0:
            asks.append((best_bid, max_sell))
        if cur_diff - wall_mid + best_bid - mean_diff >= thr_close_e and pos > 0:
            asks.append((best_bid, pos))
        elif cur_diff - wall_mid + best_ask - mean_diff <= -(thr_open_e + low_adj) and max_buy > 0:
            bids.append((best_ask, max_buy))
        if cur_diff - wall_mid + best_ask - mean_diff <= -thr_close_e and pos < 0:
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


def _shock_thr_scale(gamma: float, shock: bool) -> float:
    import math

    if not shock:
        return 1.0
    g = max(0.0, float(gamma)) if math.isfinite(gamma) else 0.0
    bump = SHOCK_THR_MULT * min(1.0, g * GAMMA_SCALE)
    return 1.0 + bump


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
        prev_raw = ema.get("S_prev_raw")
        shock = (
            prev_raw is not None
            and math.isfinite(float(prev_raw))
            and abs(float(S) - float(prev_raw)) >= EXTRACT_SHOCK_ABS
        )
        ema["S_prev_raw"] = float(S)

        joint_tight = _joint_tight_gate(depths)
        book_mult = JOINT_TIGHT_THR_MULT if joint_tight else WIDE_REGIME_THR_MULT

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
            spread = int(ba2) - int(bb2)
            if spread < int(STRIKE_SPREAD_FLOOR.get(k, MIN_SPREAD)):
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
            gam = bs_gamma(S_fair, float(k), T, sig)
            sc = _shock_thr_scale(gam, shock) * book_mult
            thr_o = THR_OPEN * sc
            thr_c = THR_CLOSE * sc
            pos = int((getattr(state, "position", None) or {}).get(sym, 0))
            bids, asks = _scalp_orders(
                float(wm2),
                int(bb2),
                int(ba2),
                pos,
                LIMIT_VEV,
                cur_diff,
                mean_diff,
                switch_mean,
                vega,
                thr_o,
                thr_c,
            )
            cap = WING_CAP if k in WING and vega < 0.55 else LIMIT_VEV
            if k == 5000:
                cap = min(cap, STRIKE_5000_CAP)
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
