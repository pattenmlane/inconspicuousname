"""
Frankfurt IV scalp trader driven by IV_GRID_CONFIG_PATH (JSON), for grid backtests.

Each TestRunner day: config must contain csv_day (0/1/2), coeffs, IV_SCALPING_THR,
TARGET_VOUCHER, K_STRIKE, t_kind (om_wind|om_nowind|ti_wind|ti_nowind|tm_wind|tm_nowind).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_MA = _REPO / "round3work/plotting/original_method/wind_down/combined_analysis"
_MA_NW = _REPO / "round3work/plotting/original_method/no_wind_down/combined_analysis"
_NB_W = _REPO / "round3work/plotting/test_implementation/wind_down/nb_method_core.py"
_NB_NW = _REPO / "round3work/plotting/test_implementation/no_wind_down/nb_method_core.py"

for p in (_MA, _MA_NW):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from frankfurt_iv_scalp_core import (  # noqa: E402
    book_from_order_depth,
    compute_option_indicators,
    get_iv_scalping_orders_frankfurt,
    load_calibration,
    synthetic_walls_if_missing,
)

_CAL_PATH = Path(__file__).resolve().parent / "calibration.json"
_EMA_KEY = "ema_iv_grid"
MAF_BID = 0

_piv_w: Any = None
_piv_nw: Any = None
_nb_w: Any = None
_nb_nw: Any = None
_nb_cache: dict[tuple[str, int], tuple[dict[int, int], int, Any]] = {}


def _load_plot_iv(combined_dir: Path) -> Any:
    p = combined_dir / "plot_iv_smile_round3.py"
    name = f"piv_{abs(hash(str(p))) % 1_000_000}"
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise RuntimeError(str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load_nb(nb_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"nb_{nb_path.parent.name}", nb_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(str(nb_path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _piv(kind: str) -> Any:
    global _piv_w, _piv_nw
    if kind == "om_wind":
        if _piv_w is None:
            _piv_w = _load_plot_iv(_MA.resolve())
        return _piv_w
    if kind == "om_nowind":
        if _piv_nw is None:
            _piv_nw = _load_plot_iv(_MA_NW.resolve())
        return _piv_nw
    raise ValueError(kind)


def _nb(kind: str) -> Any:
    global _nb_w, _nb_nw
    if kind == "ti_wind":
        if _nb_w is None:
            _nb_w = _load_nb(_NB_W.resolve())
        return _nb_w
    if kind == "ti_nowind":
        if _nb_nw is None:
            _nb_nw = _load_nb(_NB_NW.resolve())
        return _nb_nw
    raise ValueError(kind)


def _nb_mp_d0(t_kind: str, csv_day: int) -> tuple[dict[int, int], int, Any]:
    key = (t_kind, csv_day)
    if key in _nb_cache:
        return _nb_cache[key]
    nb = _nb("ti_wind" if t_kind == "ti_wind" else "ti_nowind")
    wf = nb.load_day_wide(csv_day).sort_index()
    mp = nb.index_map_timestamp_to_row_idx(wf)
    d0 = int(nb.dte_from_csv_day(csv_day))
    _nb_cache[key] = (mp, d0, nb)
    return _nb_cache[key]


def _t_tm(ts: int, wind: bool) -> float:
    prog = (int(ts) // 100) / 10_000.0
    d0 = 5.0
    d_eff = max(d0 - prog, 1e-6) if wind else d0
    return float(d_eff / 365.0)


def t_years_for_kind(t_kind: str, csv_day: int, ts: int) -> float:
    ts_i = int(ts)
    if t_kind in ("om_wind", "om_nowind"):
        piv = _piv(t_kind)
        return float(piv.t_years_effective(int(csv_day), ts_i))
    if t_kind in ("ti_wind", "ti_nowind"):
        mp, d0, nb = _nb_mp_d0(t_kind, int(csv_day))
        if ts_i not in mp:
            return float("nan")
        t_idx = mp[ts_i]
        return float(nb.expiration_time_years(d0, t_idx))
    if t_kind == "tm_wind":
        return _t_tm(ts_i, True)
    if t_kind == "tm_nowind":
        return _t_tm(ts_i, False)
    raise ValueError(t_kind)


def model_id_to_t_kind(model_id: str) -> str:
    if model_id.startswith("original_method"):
        return "om_wind" if "wind_down" in model_id else "om_nowind"
    if model_id.startswith("test_implementation"):
        return "ti_wind" if "wind_down" in model_id else "ti_nowind"
    if model_id.startswith("truemethod"):
        return "tm_wind" if "wind_down" in model_id else "tm_nowind"
    raise ValueError(model_id)


def _cfg() -> dict[str, Any]:
    path = os.environ.get("IV_GRID_CONFIG_PATH", "").strip()
    if not path:
        raise RuntimeError("Set IV_GRID_CONFIG_PATH to grid cell JSON")
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


class Trader:
    def run(self, state: TradingState):
        cfg = _cfg()
        cal = load_calibration(_CAL_PATH)
        cal["coeffs_high_to_low"] = list(cfg["coeffs_high_to_low"])
        cal["IV_SCALPING_THR"] = float(cfg["IV_SCALPING_THR"])
        for k in ("THR_OPEN", "THR_CLOSE", "LOW_VEGA_THR_ADJ", "THEO_NORM_WINDOW", "IV_SCALPING_WINDOW", "LOW_VEGA_CUTOFF", "POSITION_LIMIT"):
            if k in cfg:
                cal[k] = cfg[k]

        opt = str(cfg["TARGET_VOUCHER"])
        K = int(cfg["K_STRIKE"])
        u = str(cfg.get("UNDERLYING", "VELVETFRUIT_EXTRACT"))
        csv_day = int(cfg["csv_day"])
        t_kind = str(cfg.get("t_kind") or model_id_to_t_kind(str(cfg["model_id"])))

        sym_o = _symbol_for_product(state, opt)
        sym_u = _symbol_for_product(state, u)
        store = _parse_td(getattr(state, "traderData", None))
        ema_store: dict[str, float] = store.get(_EMA_KEY) if isinstance(store.get(_EMA_KEY), dict) else {}
        if not isinstance(ema_store, dict):
            ema_store = {}
        ema_store = {str(k): float(v) for k, v in ema_store.items() if isinstance(v, (int, float))}

        if sym_o is None or sym_u is None:
            store[_EMA_KEY] = ema_store
            return {}, 0, json.dumps(store, separators=(",", ":"))

        depths = getattr(state, "order_depths", None) or {}
        depth_o: OrderDepth | None = depths.get(sym_o)
        depth_u: OrderDepth | None = depths.get(sym_u)
        if depth_o is None or depth_u is None:
            store[_EMA_KEY] = ema_store
            return {}, 0, json.dumps(store, separators=(",", ":"))

        _, _, bid_w, ask_w, bb, ba, wm = book_from_order_depth(depth_o)
        _, _, _, _, ubb, uba, _ = book_from_order_depth(depth_u)
        if ubb is None or uba is None:
            store[_EMA_KEY] = ema_store
            return {}, 0, json.dumps(store, separators=(",", ":"))

        u_mid = 0.5 * float(ubb) + 0.5 * float(uba)
        bid_w2, ask_w2, wm2, bb2, ba2 = synthetic_walls_if_missing(bid_w, ask_w, bb, ba)
        if wm2 is None or bb2 is None or ba2 is None:
            store[_EMA_KEY] = ema_store
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ts = int(getattr(state, "timestamp", 0))
        warmup = int(cal.get("WARMUP_TS_DIV100", 10))
        if ts // 100 < warmup:
            store[_EMA_KEY] = ema_store
            return {}, 0, json.dumps(store, separators=(",", ":"))

        T = t_years_for_kind(t_kind, csv_day, ts)
        if not math.isfinite(T) or T <= 0:
            store[_EMA_KEY] = ema_store
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ind = compute_option_indicators(cal, ema_store, u_mid, K, T, float(wm2), float(bb2), float(ba2), opt)

        pos_sym = int((getattr(state, "position", None) or {}).get(sym_o, 0))
        lim = int(cal["POSITION_LIMIT"])
        max_buy = lim - pos_sym
        max_sell = lim + pos_sym

        bids, asks = get_iv_scalping_orders_frankfurt(
            cal, ind, float(wm2), float(bb2), float(ba2), pos_sym, max_buy, max_sell
        )

        def _merge(levels: list[tuple[int, int]]) -> dict[int, int]:
            m: dict[int, int] = {}
            for px, q in levels:
                m[int(px)] = m.get(int(px), 0) + int(q)
            return m

        orders: list[Order] = []
        for px, q in _merge(asks).items():
            q = min(int(q), lim + pos_sym)
            if q > 0:
                orders.append(Order(sym_o, int(px), -q))
        for px, q in _merge(bids).items():
            q = min(int(q), lim - pos_sym)
            if q > 0:
                orders.append(Order(sym_o, int(px), q))

        store[_EMA_KEY] = ema_store
        return {sym_o: orders}, 0, json.dumps(store, separators=(",", ":"))
