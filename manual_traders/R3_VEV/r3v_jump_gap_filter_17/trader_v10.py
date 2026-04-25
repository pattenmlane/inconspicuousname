"""
r3v_jump_gap_filter_17 — iteration 10

Parent: v8 (best hybrid so far): jump pause + neighbor compression + nearest-ATM
theo_diff scalp; hydrogel-only while paused.

New orthogonal risk control from tape analysis (analysis_time_bucket_dev_vega.json):
Q4 (last session quartile) has larger |wall_mid - theo| tails and lower ATM vega.
Rule: do not open new VEV risk after _VEV_ENTRY_CUTOFF_FRAC session progress.
Still allow existing positions to reduce when signal flips.

TTE: t_years_effective(csv_day, ts); csv_day inferred from first extract mid.
"""
from __future__ import annotations

import json
import math
import sys
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
    "THEO_NORM_WINDOW": 20,
    "IV_SCALPING_WINDOW": 100,
    "POSITION_LIMIT": 300,
    "UNDERLYING": "VELVETFRUIT_EXTRACT",
    "WARMUP_TS_DIV100": 10,
}

_TD_KEY = "r3v17"
_JUMP_DS = 3.0
_PAUSE_UNTIL_TICK = 120
_NEIGH_RESID_MAX = 8.0
_ATM_STRIKE_BAND = 350
_SCALP_THR = 0.42
_LOW_VEGA = 0.9
_MAX_VEV_QTY = 22
_HYDRO_ORDER_SIZE = 15
_VEV_ENTRY_CUTOFF_FRAC = 0.75


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


class Trader:
    def run(self, state: TradingState):
        cal = _load_cal()
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

        pause_until = int(bucket.get("pause_until_tick", -1))
        if dS >= _JUMP_DS:
            pause_until = max(pause_until, tick + _PAUSE_UNTIL_TICK)
        bucket["pause_until_tick"] = pause_until
        paused = tick < pause_until

        warmup = int(cal.get("WARMUP_TS_DIV100", 10))
        T_base = t_years_effective(csv_day, ts)

        lim_v = int(cal["POSITION_LIMIT"])
        lim_h = 200

        orders_out: dict[str, list[Order]] = {}

        coeffs = cal["coeffs_high_to_low"]
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
            k_star = min(STRIKES, key=lambda kk: abs(float(kk) - u_mid))
            if abs(float(k_star) - u_mid) <= _ATM_STRIKE_BAND + 150:
                sym_o = _symbol_for_product(state, f"VEV_{k_star}")
                if sym_o is not None:
                    depth_o = depths.get(sym_o)
                    if depth_o is not None:
                        _, _, bid_w, ask_w, bb, ba, wm = book_from_order_depth(depth_o)
                        _, _, wm2, bb2, ba2 = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
                        if wm2 is not None and bb2 is not None and ba2 is not None:
                            ind = compute_option_indicators(
                                cal, ema_store, u_mid, k_star, T_base, float(wm2), float(bb2), float(ba2), sym_o
                            )
                            ctd = ind.get("current_theo_diff")
                            mtd = ind.get("mean_theo_diff")
                            vega = float(ind.get("vega") or 0.0)
                            if ctd is not None and mtd is not None and vega >= _LOW_VEGA:
                                sell_px = max(int(bb2) - 1, 1)
                                buy_px = int(ba2) + 1
                                pos_sym = int(pos.get(sym_o, 0))
                                dev = float(ctd) - float(mtd)
                                qfrac = tick / 10000.0
                                allow_new_entries = qfrac <= _VEV_ENTRY_CUTOFF_FRAC
                                if dev > _SCALP_THR:
                                    if pos_sym > 0:
                                        q = min(pos_sym, _MAX_VEV_QTY)
                                        if q > 0:
                                            orders_out[sym_o] = [Order(sym_o, sell_px, -q)]
                                    elif allow_new_entries and pos_sym > -lim_v + 5:
                                        q = min(_MAX_VEV_QTY, lim_v + pos_sym)
                                        if q > 0:
                                            orders_out[sym_o] = [Order(sym_o, sell_px, -q)]
                                elif dev < -_SCALP_THR:
                                    if pos_sym < 0:
                                        q = min(-pos_sym, _MAX_VEV_QTY)
                                        if q > 0:
                                            orders_out[sym_o] = [Order(sym_o, buy_px, q)]
                                    elif allow_new_entries and pos_sym < lim_v - 5:
                                        q = min(_MAX_VEV_QTY, lim_v - pos_sym)
                                        if q > 0:
                                            orders_out[sym_o] = [Order(sym_o, buy_px, q)]

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
