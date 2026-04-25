"""
Velvet spike playbook v10: v7 + targeted burst strike exclusion (VEV_4500).

- Keep same ATM bands and burst lot cap as v7.
- In burst only, skip VEV_4500 (worst burst proxy among active strikes in analysis)
  while still allowing VEV_4000 and ATM/upside ring.
- Calm and low-z behavior unchanged from v7.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

_REPO = Path(__file__).resolve().parents[3]
_5200 = _REPO / "round3work" / "voucher_work" / "5200_work"
_MA = _REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"
for p in (_5200, _MA):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from frankfurt_iv_scalp_core import (  # noqa: E402
    book_from_order_depth,
    compute_option_indicators,
    get_iv_scalping_orders_frankfurt,
    load_calibration,
    synthetic_walls_if_missing,
)

_CAL_PATH = _5200 / "calibration.json"
_EMA_KEY = "ema_r3v_velvet_spike"

IV_THR_BASE = 0.44
BURST_IV_MULT = 1.02
CALM_IV_MULT = 0.98
Z_WINDOW = 50
Z_SPIKE = 2.2
ABS_DS_SPIKE = 7.0
BURST_OPEN_MULT = 0.88
CALM_OPEN_MULT = 1.06
WARMUP_DIV100 = 10
ATM_REL_BURST = 0.24
ATM_REL_CALM = 0.10
LOW_Z_CALM = 0.85
BURST_MAX_LOT = 80
BURST_EXCLUDE_STRIKES = {4500}

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

_EMBEDDED_CAL: dict[str, Any] = {
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
    "WARMUP_TS_DIV100": WARMUP_DIV100,
}


def _load_plot_iv() -> Any:
    p = _MA / "plot_iv_smile_round3.py"
    name = "piv_r3v_velvet_v10"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_piv_mod: Any = None


def t_years(csv_day: int, ts: int) -> float:
    global _piv_mod
    if _piv_mod is None:
        _piv_mod = _load_plot_iv()
    return float(_piv_mod.t_years_effective(int(csv_day), int(ts)))


def _load_cal() -> dict[str, Any]:
    if _CAL_PATH.is_file():
        return load_calibration(_CAL_PATH)
    return dict(_EMBEDDED_CAL)


def _symbol_for_product(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _rolling_z(series: list[float], window: int) -> float:
    if len(series) < max(window, 5):
        return 0.0
    w = series[-window:]
    m = sum(w) / len(w)
    var = sum((x - m) ** 2 for x in w) / max(len(w) - 1, 1)
    std = math.sqrt(var) + 1e-9
    return abs(w[-1] - m) / std


def _strike_in_cluster(S: float, K: int, rel: float) -> bool:
    if S <= 0:
        return False
    return abs(float(K) / S - 1.0) <= rel


class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        cal = _load_cal()
        csv_day = int(getattr(state, "csv_day", 0))
        store = _parse_td(getattr(state, "traderData", None))
        ema_store: dict[str, float] = store.get(_EMA_KEY) if isinstance(store.get(_EMA_KEY), dict) else {}
        if not isinstance(ema_store, dict):
            ema_store = {}
        ema_store = {str(k): float(v) for k, v in ema_store.items() if isinstance(v, (int, float))}

        hist = store.get("s_hist")
        if not isinstance(hist, list):
            hist = []
        hist = [float(x) for x in hist if isinstance(x, (int, float))][-120:]

        sym_u = _symbol_for_product(state, "VELVETFRUIT_EXTRACT")
        depths = getattr(state, "order_depths", None) or {}
        if sym_u is None:
            store[_EMA_KEY] = ema_store
            store["s_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        depth_u = depths.get(sym_u)
        if depth_u is None:
            store[_EMA_KEY] = ema_store
            store["s_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        _, _, _, _, ubb, uba, _ = book_from_order_depth(depth_u)
        if ubb is None or uba is None:
            store[_EMA_KEY] = ema_store
            store["s_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        S = 0.5 * float(ubb) + 0.5 * float(uba)
        abs_dS = abs(S - hist[-1]) if hist else 0.0
        hist.append(S)
        hist = hist[-120:]

        dlog_hist = store.get("dlog_hist")
        if not isinstance(dlog_hist, list):
            dlog_hist = []
        dlog_hist = [float(x) for x in dlog_hist if isinstance(x, (int, float))][-200:]
        if len(hist) >= 2:
            prev = hist[-2]
            if prev > 0:
                dlog_hist.append(math.log(S / prev))
        dlog_hist = dlog_hist[-200:]
        z = _rolling_z(dlog_hist, Z_WINDOW)
        burst = z >= Z_SPIKE or abs_dS >= ABS_DS_SPIKE
        atm_rel = ATM_REL_BURST if burst else ATM_REL_CALM
        very_calm = (not burst) and (z <= LOW_Z_CALM)

        ts = int(getattr(state, "timestamp", 0))
        if ts // 100 < WARMUP_DIV100:
            store[_EMA_KEY] = ema_store
            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        T = t_years(csv_day, ts)
        if not math.isfinite(T) or T <= 0:
            store[_EMA_KEY] = ema_store
            store["s_hist"] = hist
            store["dlog_hist"] = dlog_hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        base_open = float(cal["THR_OPEN"])
        thr_open = base_open * (BURST_OPEN_MULT if burst else CALM_OPEN_MULT)
        cal_use = dict(cal)
        cal_use["THR_OPEN"] = thr_open
        cal_use["IV_SCALPING_THR"] = max(
            0.25, float(IV_THR_BASE) * (BURST_IV_MULT if burst else CALM_IV_MULT)
        )

        pos = getattr(state, "position", None) or {}
        orders_out: dict[str, list[Order]] = {}

        for strike in STRIKES:
            if not _strike_in_cluster(S, strike, atm_rel):
                continue
            if burst and strike in BURST_EXCLUDE_STRIKES:
                continue
            opt = f"VEV_{strike}"
            sym_o = _symbol_for_product(state, opt)
            if sym_o is None:
                continue
            depth_o = depths.get(sym_o)
            if depth_o is None:
                continue
            _, _, bid_w, ask_w, bb, ba, wm = book_from_order_depth(depth_o)
            bid_w2, ask_w2, wm2, bb2, ba2 = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
            if wm2 is None or bb2 is None or ba2 is None:
                continue

            ind = compute_option_indicators(
                cal_use, ema_store, S, strike, T, float(wm2), float(bb2), float(ba2), opt
            )
            pos_sym = int(pos.get(sym_o, 0))
            lim = int(cal.get("POSITION_LIMIT", 300))
            bids, asks = get_iv_scalping_orders_frankfurt(
                cal_use, ind, float(wm2), float(bb2), float(ba2), pos_sym, lim - pos_sym, lim + pos_sym
            )
            if very_calm:
                # Mean-revert fallback: in very calm regime, avoid opening fresh inventory.
                if pos_sym > 0:
                    bids = []
                elif pos_sym < 0:
                    asks = []
                else:
                    bids = []
                    asks = []
            lot_cap = int(BURST_MAX_LOT) if burst else 10_000
            ol: list[Order] = []
            for px, q in asks:
                q = min(int(q), lim + pos_sym, lot_cap)
                if q > 0:
                    ol.append(Order(sym_o, int(px), -q))
            for px, q in bids:
                q = min(int(q), lim - pos_sym, lot_cap)
                if q > 0:
                    ol.append(Order(sym_o, int(px), q))
            if ol:
                orders_out[sym_o] = ol

        store[_EMA_KEY] = ema_store
        store["s_hist"] = hist
        store["dlog_hist"] = dlog_hist
        store["last_burst"] = bool(burst)
        store["atm_rel"] = atm_rel
        return orders_out, 0, json.dumps(store, separators=(",", ":"))
