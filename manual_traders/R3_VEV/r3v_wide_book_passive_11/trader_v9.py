"""v9 (family 13 realized_vol_regime): RV-IV execution regime.

Regime signal: rv_ema - iv_ema where
- rv_ema: EMA of squared log-returns of VELVETFRUIT_EXTRACT mid (tick realized vol proxy)
- iv_ema: EMA of average BS implied vol from VEV_4000/VEV_4500 mids

Policy split (minimal model side):
- stressed regime: passive-only + wider quotes
- calm regime: allow small aggressive fills when edge vs fair is strong

Core wide-book passive thesis on VEV_4000/4500 is retained.
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Dict, List, Optional

from datamodel import Order, OrderDepth, TradingState

_N = NormalDist()

UNDER = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
LIMITS: Dict[str, int] = {
    HYDRO: 200,
    UNDER: 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}
STRIKE = {"VEV_4000": 4000, "VEV_4500": 4500}
TARGET_VEV = ("VEV_4000", "VEV_4500")
TTE_DAYS_PROXY = 6
DAYS_PER_YEAR = 365

WIDE_SPREAD_MIN = 15
DEEP_OFFSET = {"VEV_4000": 4, "VEV_4500": 5}
PASSIVE_SIZE = 3

EXTRACT_EMA_ALPHA = 0.02
EXTRACT_PASSIVE_SIZE = 6
HYDRO_EMA_ALPHA = 0.02
HYDRO_PASSIVE_SIZE = 3
HYDRO_WIDE_MIN = 15

DELTA_SKEW = 0.3
VEGA_SKEW = 0.0005
SKEW_CAP = 8

# Regime controls
RV_ALPHA = 0.06
IV_ALPHA = 0.06
RV_IV_STRESS_CUT = -0.50
CALM_AGGR_EDGE_EXTRACT = 4.0
CALM_AGGR_EDGE_HYDRO = 7.0
STRESS_WIDE_PAD_EXTRACT = 2
STRESS_WIDE_PAD_HYDRO = 2
CALM_AGGR_SIZE_EXTRACT = 2
CALM_AGGR_SIZE_HYDRO = 2


def _pos(state: TradingState, sym: str) -> int:
    return int(state.position.get(sym, 0))


def _mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return (bb + ba) / 2.0


def _best_bid_ask(od: OrderDepth) -> tuple[Optional[int], Optional[int]]:
    if not od.buy_orders or not od.sell_orders:
        return None, None
    return max(od.buy_orders.keys()), min(od.sell_orders.keys())


def _spread(od: OrderDepth) -> Optional[int]:
    bb, ba = _best_bid_ask(od)
    if bb is None or ba is None:
        return None
    return ba - bb


def bs_call_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return _N.cdf(d1)


def bs_call_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    return S * _N.pdf(d1) * math.sqrt(T)


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return S * _N.cdf(d1) - K * _N.cdf(d2)


def bs_call_iv(S: float, K: float, T: float, mid: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if mid <= intrinsic + 1e-6:
        return None
    lo, hi = 1e-6, 5.0
    for _ in range(55):
        m = 0.5 * (lo + hi)
        if bs_call_price(S, K, T, m) > mid:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


class Trader:
    def run(self, state: TradingState):
        data: dict = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except json.JSONDecodeError:
                data = {}

        orders: Dict[str, List[Order]] = {p: [] for p in LIMITS}

        od_u = state.order_depths.get(UNDER)
        mid_u = _mid(od_u) if od_u else None
        e_ema = data.get("e_ema")
        if mid_u is not None:
            if e_ema is None:
                e_ema = mid_u
            else:
                e_ema = (1.0 - EXTRACT_EMA_ALPHA) * float(e_ema) + EXTRACT_EMA_ALPHA * mid_u
        data["e_ema"] = e_ema

        od_h = state.order_depths.get(HYDRO)
        mid_h = _mid(od_h) if od_h else None
        h_ema = data.get("h_ema")
        if mid_h is not None:
            if h_ema is None:
                h_ema = mid_h
            else:
                h_ema = (1.0 - HYDRO_EMA_ALPHA) * float(h_ema) + HYDRO_EMA_ALPHA * mid_h
        data["h_ema"] = h_ema

        # RV proxy from underlying returns
        prev_mid_u = data.get("prev_mid_u")
        rv_ema = data.get("rv_ema", 0.0)
        if mid_u is not None and prev_mid_u and prev_mid_u > 0:
            ret = math.log(mid_u / float(prev_mid_u))
            r2 = ret * ret
            rv_ema = (1.0 - RV_ALPHA) * float(rv_ema) + RV_ALPHA * r2
        data["prev_mid_u"] = mid_u

        T = TTE_DAYS_PROXY / DAYS_PER_YEAR
        vev_delta: Dict[str, float] = {}
        vev_vega: Dict[str, float] = {}
        iv_vals: List[float] = []
        if mid_u is not None:
            for sym in TARGET_VEV:
                od = state.order_depths.get(sym)
                if od is None:
                    continue
                m = _mid(od)
                if m is None:
                    continue
                K = STRIKE[sym]
                iv = bs_call_iv(mid_u, K, T, m)
                if iv is not None:
                    iv_vals.append(iv)
                    vev_delta[sym] = bs_call_delta(mid_u, K, T, iv)
                    vev_vega[sym] = bs_call_vega(mid_u, K, T, iv)
                else:
                    vev_delta[sym] = 0.0
                    vev_vega[sym] = 0.0

        iv_now = sum(iv_vals) / len(iv_vals) if iv_vals else data.get("iv_ema")
        iv_ema = data.get("iv_ema", iv_now if iv_now is not None else 0.25)
        if iv_now is not None:
            iv_ema = (1.0 - IV_ALPHA) * float(iv_ema) + IV_ALPHA * float(iv_now)

        data["rv_ema"] = rv_ema
        data["iv_ema"] = iv_ema
        rv_level = math.sqrt(max(float(rv_ema), 0.0)) if rv_ema is not None else 0.0
        rv_iv_diff = rv_level - float(iv_ema)
        stressed = rv_iv_diff > RV_IV_STRESS_CUT

        net_call_delta = sum(_pos(state, s) * vev_delta.get(s, 0.0) for s in TARGET_VEV)
        net_opt_vega = sum(_pos(state, s) * vev_vega.get(s, 0.0) for s in TARGET_VEV)
        skew = int(round(net_call_delta * DELTA_SKEW + net_opt_vega * VEGA_SKEW))
        skew = max(-SKEW_CAP, min(SKEW_CAP, skew))

        # Core passive voucher quotes; stressed widens one extra tick.
        for sym in TARGET_VEV:
            od = state.order_depths.get(sym)
            if od is None:
                continue
            sp = _spread(od)
            if sp is None or sp < WIDE_SPREAD_MIN:
                continue
            bb, ba = _best_bid_ask(od)
            if bb is None or ba is None:
                continue
            pos = _pos(state, sym)
            lim = LIMITS[sym]
            off = DEEP_OFFSET[sym] + (1 if stressed else 0)
            bid_px = bb + off
            ask_px = ba - off
            if bid_px >= ask_px:
                continue
            buy_room = lim - pos
            sell_room = lim + pos
            q_base = PASSIVE_SIZE if not stressed else max(1, PASSIVE_SIZE - 1)
            if buy_room > 0:
                orders[sym].append(Order(sym, int(bid_px), min(q_base, buy_room)))
            if sell_room > 0:
                orders[sym].append(Order(sym, int(ask_px), -min(q_base, sell_room)))

        if od_u is not None and e_ema is not None:
            bb, ba = _best_bid_ask(od_u)
            if bb is not None and ba is not None:
                pos = _pos(state, UNDER)
                lim = LIMITS[UNDER]
                fair = float(e_ema)
                buy_room = lim - pos
                sell_room = lim + pos

                if stressed:
                    bid_px = min(bb, int(math.floor(fair)) - STRESS_WIDE_PAD_EXTRACT - skew)
                    ask_px = max(ba, int(math.ceil(fair)) + STRESS_WIDE_PAD_EXTRACT - skew)
                    if buy_room > 0 and bid_px > 0 and bid_px < ba:
                        orders[UNDER].append(Order(UNDER, bid_px, min(EXTRACT_PASSIVE_SIZE, buy_room)))
                    if sell_room > 0 and ask_px > bb:
                        orders[UNDER].append(Order(UNDER, ask_px, -min(EXTRACT_PASSIVE_SIZE, sell_room)))
                else:
                    bid_px = min(bb + 1, int(math.floor(fair)) - 1 - skew)
                    ask_px = max(ba - 1, int(math.ceil(fair)) + 1 - skew)
                    if buy_room > 0 and bid_px > 0 and bid_px < ba:
                        orders[UNDER].append(Order(UNDER, bid_px, min(EXTRACT_PASSIVE_SIZE, buy_room)))
                    if sell_room > 0 and ask_px > bb:
                        orders[UNDER].append(Order(UNDER, ask_px, -min(EXTRACT_PASSIVE_SIZE, sell_room)))

                    # Calm regime: allow small aggressive fills when edge is strong.
                    if buy_room > 0 and (fair - ba) >= CALM_AGGR_EDGE_EXTRACT:
                        orders[UNDER].append(Order(UNDER, ba, min(CALM_AGGR_SIZE_EXTRACT, buy_room)))
                    if sell_room > 0 and (bb - fair) >= CALM_AGGR_EDGE_EXTRACT:
                        orders[UNDER].append(Order(UNDER, bb, -min(CALM_AGGR_SIZE_EXTRACT, sell_room)))

        if od_h is not None and h_ema is not None:
            sp = _spread(od_h)
            if sp is not None and sp >= HYDRO_WIDE_MIN:
                bb, ba = _best_bid_ask(od_h)
                if bb is not None and ba is not None:
                    pos = _pos(state, HYDRO)
                    lim = LIMITS[HYDRO]
                    fair = float(h_ema)
                    buy_room = lim - pos
                    sell_room = lim + pos

                    if stressed:
                        bid_px = min(bb, int(math.floor(fair)) - STRESS_WIDE_PAD_HYDRO)
                        ask_px = max(ba, int(math.ceil(fair)) + STRESS_WIDE_PAD_HYDRO)
                        if buy_room > 0 and bid_px > 0 and bid_px < ba:
                            orders[HYDRO].append(Order(HYDRO, bid_px, min(HYDRO_PASSIVE_SIZE, buy_room)))
                        if sell_room > 0 and ask_px > bb:
                            orders[HYDRO].append(Order(HYDRO, ask_px, -min(HYDRO_PASSIVE_SIZE, sell_room)))
                    else:
                        bid_px = min(bb + 1, int(math.floor(fair)) - 1)
                        ask_px = max(ba - 1, int(math.ceil(fair)) + 1)
                        if buy_room > 0 and bid_px > 0 and bid_px < ba:
                            orders[HYDRO].append(Order(HYDRO, bid_px, min(HYDRO_PASSIVE_SIZE, buy_room)))
                        if sell_room > 0 and ask_px > bb:
                            orders[HYDRO].append(Order(HYDRO, ask_px, -min(HYDRO_PASSIVE_SIZE, sell_room)))

                        if buy_room > 0 and (fair - ba) >= CALM_AGGR_EDGE_HYDRO:
                            orders[HYDRO].append(Order(HYDRO, ba, min(CALM_AGGR_SIZE_HYDRO, buy_room)))
                        if sell_room > 0 and (bb - fair) >= CALM_AGGR_EDGE_HYDRO:
                            orders[HYDRO].append(Order(HYDRO, bb, -min(CALM_AGGR_SIZE_HYDRO, sell_room)))

        data["regime"] = "stressed" if stressed else "calm"
        data["rv_iv_diff"] = rv_iv_diff
        return orders, 0, json.dumps(data)
