"""
r3v_jump_gap_filter_17 — iteration 2

Grid vs v1: (1) slightly more permissive neighbor residual cap, (2) looser IV-scalp
thresholds in a cal copy, (3) one-tick more aggressive limit prices so book + worse
fills can execute (sell one tick below best bid, buy one tick above best ask).

Same jump cooldown (|ΔS|) + pause hydrogel MM as v1.
"""
from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

REPO = Path(__file__).resolve().parents[3]
_VW = REPO / "round3work" / "voucher_work" / "5200_work"
_CA = REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"
for p in (_VW, _CA):
    if p.is_dir():
        sys.path.insert(0, str(p))

from frankfurt_iv_scalp_core import (  # noqa: E402
    book_from_order_depth,
    compute_option_indicators,
    get_iv_scalping_orders_frankfurt,
    get_option_values,
    load_calibration,
    synthetic_walls_if_missing,
)
from plot_iv_smile_round3 import t_years_effective  # noqa: E402

CAL_PATH = _VW / "calibration.json"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
U_PRODUCT = "VELVETFRUIT_EXTRACT"
H_PRODUCT = "HYDROGEL_PACK"

_EMBEDDED_CAL: dict[str, Any] = {
    "dte_at_open_by_csv_day": {"0": 8, "1": 7, "2": 6},
    "coeffs_high_to_low": [0.14215151147708086, -0.0016298611395181932, 0.23576325646627055],
    "THR_OPEN": 0.5,
    "THR_CLOSE": 0.0,
    "LOW_VEGA_THR_ADJ": 0.5,
    "THEO_NORM_WINDOW": 20,
    "IV_SCALPING_WINDOW": 100,
    "IV_SCALPING_THR": 0.7,
    "LOW_VEGA_CUTOFF": 1.0,
    "POSITION_LIMIT": 300,
    "UNDERLYING": "VELVETFRUIT_EXTRACT",
    "WARMUP_TS_DIV100": 10,
}

_TD_KEY = "r3v17"
_EXTRACT_EMA_W = 400
_JUMP_DS = 3.0
_PAUSE_UNTIL_TICK = 120
_NEIGH_RESID_MAX = 10.0
_ATM_STRIKE_BAND = 350
_HYDRO_ORDER_SIZE = 15


def _first_extract_mid(csv_day: int) -> float:
    import pandas as pd

    p = REPO / "Prosperity4Data" / "ROUND_3" / f"prices_round_3_day_{csv_day}.csv"
    df = pd.read_csv(p, sep=";", nrows=500)
    row = df[df["product"] == U_PRODUCT].iloc[0]
    return float(row["mid_price"])


_OPEN_S_PROBE = [_first_extract_mid(d) for d in (0, 1, 2)]


def _symbol_for_product(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _load_cal() -> dict[str, Any]:
    if CAL_PATH.exists():
        return load_calibration(CAL_PATH)
    return dict(_EMBEDDED_CAL)


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _ema(prev: float, x: float, window: int) -> float:
    alpha = 2.0 / (float(window) + 1.0)
    return alpha * x + (1.0 - alpha) * prev


def _merge_orders(olist: list[Order]) -> list[Order]:
    m: dict[tuple[str, int], int] = {}
    for o in olist:
        k = (o.symbol, o.price)
        m[k] = m.get(k, 0) + int(o.quantity)
    return [Order(sym, px, q) for (sym, px), q in m.items() if q != 0]


def _cal_for_vev(base: dict[str, Any]) -> dict[str, Any]:
    c = deepcopy(base)
    c["THR_OPEN"] = 0.35
    c["IV_SCALPING_THR"] = 0.55
    return c


class Trader:
    def run(self, state: TradingState):
        cal_base = _load_cal()
        cal_vev = _cal_for_vev(cal_base)
        store = _parse_td(getattr(state, "traderData", None))
        bucket: dict[str, Any] = store.get(_TD_KEY) if isinstance(store.get(_TD_KEY), dict) else {}
        if not isinstance(bucket, dict):
            bucket = {}
        ema_store: dict[str, float] = bucket.get("ema") if isinstance(bucket.get("ema"), dict) else {}
        if not isinstance(ema_store, dict):
            ema_store = {}
        ema_store = {str(k): float(v) for k, v in ema_store.items() if isinstance(v, (int, float))}

        sym_u = _symbol_for_product(state, U_PRODUCT)
        sym_h = _symbol_for_product(state, H_PRODUCT)
        depths = getattr(state, "order_depths", None) or {}
        pos = getattr(state, "position", None) or {}

        if sym_u is None:
            bucket["ema"] = ema_store
            store[_TD_KEY] = bucket
            return {}, 0, json.dumps(store, separators=(",", ":"))

        depth_u: OrderDepth | None = depths.get(sym_u)
        if depth_u is None:
            bucket["ema"] = ema_store
            store[_TD_KEY] = bucket
            return {}, 0, json.dumps(store, separators=(",", ":"))

        _, _, _, _, ubb, uba, _ = book_from_order_depth(depth_u)
        if ubb is None or uba is None:
            bucket["ema"] = ema_store
            store[_TD_KEY] = bucket
            return {}, 0, json.dumps(store, separators=(",", ":"))

        u_mid = 0.5 * float(ubb) + 0.5 * float(uba)
        ts = int(getattr(state, "timestamp", 0))
        tick = ts // 100

        if ts == 0 or "csv_day" not in bucket:
            d_hat = int(np.argmin([abs(u_mid - p) for p in _OPEN_S_PROBE]))
            bucket["csv_day"] = d_hat
            bucket["pause_until_tick"] = -1
        csv_day = int(bucket["csv_day"])

        prev_s = float(bucket.get("prev_S", u_mid))
        dS = abs(u_mid - prev_s) if ts > 0 else 0.0
        bucket["prev_S"] = u_mid

        s_ema = float(bucket.get("S_ema", u_mid))
        s_ema = _ema(s_ema, u_mid, _EXTRACT_EMA_W)
        bucket["S_ema"] = s_ema

        pause_until = int(bucket.get("pause_until_tick", -1))
        if dS >= _JUMP_DS:
            pause_until = max(pause_until, tick + _PAUSE_UNTIL_TICK)
        bucket["pause_until_tick"] = pause_until
        paused = tick < pause_until

        warmup = int(cal_base.get("WARMUP_TS_DIV100", 10))
        T_base = t_years_effective(csv_day, ts)

        lim_v = int(cal_base["POSITION_LIMIT"])
        lim_u = 200
        lim_h = 200

        orders_out: dict[str, list[Order]] = {}

        coeffs = cal_base["coeffs_high_to_low"]
        wall_mids: list[float] = []
        theos: list[float] = []
        for k in STRIKES:
            if abs(float(k) - u_mid) > _ATM_STRIKE_BAND:
                continue
            sym_o = _symbol_for_product(state, f"VEV_{k}")
            if sym_o is None:
                continue
            depth_o: OrderDepth | None = depths.get(sym_o)
            if depth_o is None:
                continue
            _, _, bid_w, ask_w, bb, ba, wm = book_from_order_depth(depth_o)
            _, _, wm2, bb2, ba2 = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
            if wm2 is None:
                continue
            theo, _, _ = get_option_values(float(u_mid), float(k), float(T_base), coeffs)
            if not math.isfinite(theo):
                continue
            wall_mids.append(float(wm2))
            theos.append(float(theo))
        resid_spread = 0.0
        if wall_mids and theos:
            res = [wm - th for wm, th in zip(wall_mids, theos)]
            resid_spread = float(max(res) - min(res))
        compressed = resid_spread <= _NEIGH_RESID_MAX and len(wall_mids) >= 3
        allow_v = (not paused) and compressed and tick >= warmup

        if allow_v:
            for k in STRIKES:
                if abs(float(k) - u_mid) > _ATM_STRIKE_BAND + 150:
                    continue
                sym_o = _symbol_for_product(state, f"VEV_{k}")
                if sym_o is None:
                    continue
                depth_o: OrderDepth | None = depths.get(sym_o)
                if depth_o is None:
                    continue
                _, _, bid_w, ask_w, bb, ba, wm = book_from_order_depth(depth_o)
                bid_w2, ask_w2, wm2, bb2, ba2 = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
                if wm2 is None or bb2 is None or ba2 is None:
                    continue
                ind = compute_option_indicators(
                    cal_vev, ema_store, u_mid, k, T_base, float(wm2), float(bb2), float(ba2), sym_o
                )
                pos_sym = int(pos.get(sym_o, 0))
                max_buy = lim_v - pos_sym
                max_sell = lim_v + pos_sym
                bids, asks = get_iv_scalping_orders_frankfurt(
                    cal_vev, ind, float(wm2), float(bb2), float(ba2), pos_sym, max_buy, max_sell
                )
                olist: list[Order] = []
                sell_px = max(int(bb2) - 1, 1)
                buy_px = int(ba2) + 1
                for px, q in asks:
                    q = min(int(q), lim_v + pos_sym)
                    if q > 0:
                        olist.append(Order(sym_o, sell_px, -q))
                for px, q in bids:
                    q = min(int(q), lim_v - pos_sym)
                    if q > 0:
                        olist.append(Order(sym_o, buy_px, q))
                if olist:
                    orders_out.setdefault(sym_o, []).extend(olist)

        for sym_o, ol in list(orders_out.items()):
            orders_out[sym_o] = _merge_orders(ol)

        if sym_u and paused:
            pos_u = int(pos.get(sym_u, 0))
            u_olist: list[Order] = []
            if u_mid > s_ema + 4 and pos_u > -lim_u + 15:
                u_olist.append(Order(sym_u, int(uba), -min(10, pos_u + lim_u)))
            elif u_mid < s_ema - 4 and pos_u < lim_u - 15:
                u_olist.append(Order(sym_u, int(ubb), min(10, lim_u - pos_u)))
            if u_olist:
                orders_out[sym_u] = u_olist

        if sym_h and paused and sym_h not in orders_out:
            dh = depths.get(sym_h)
            if dh is not None:
                _, _, _, _, hb, ha, _ = book_from_order_depth(dh)
                if hb is not None and ha is not None:
                    pos_h = int(pos.get(sym_h, 0))
                    spread = float(ha) - float(hb)
                    if spread >= 2.0:
                        bidp = int(hb) + 1
                        askp = int(ha) - 1
                        if bidp < askp:
                            o_h: list[Order] = []
                            if pos_h < lim_h - 5:
                                o_h.append(Order(sym_h, bidp, min(_HYDRO_ORDER_SIZE, lim_h - pos_h)))
                            if pos_h > -lim_h + 5:
                                o_h.append(Order(sym_h, askp, -min(_HYDRO_ORDER_SIZE, lim_h + pos_h)))
                            if o_h:
                                orders_out[sym_h] = o_h

        bucket["ema"] = ema_store
        store[_TD_KEY] = bucket
        return orders_out, 0, json.dumps(store, separators=(",", ":"))
