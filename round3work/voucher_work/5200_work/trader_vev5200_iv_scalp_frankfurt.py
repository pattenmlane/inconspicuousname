"""
Prosperity 4 Round 3 — VEV_5200 Frankfurt-style IV scalping (single-product upload).

Logic: frankfurt_iv_scalp_core (clone of FrankfurtHedgehogs_polished OptionTrader IV branch).

DTE mapping (historical Round 3 CSVs — hardcode competition day index to match):
  csv_day 0 → 8 DTE at open, 1 → 7, 2 → 6 (intraday winding: plot_iv_smile_round3.t_years_effective).

Run with PYTHONPATH including the backtester root (for prosperity4bt.datamodel), this folder
(core + trader), and combined_analysis, e.g.:
  PYTHONPATH="/Users/you/ProsperityRepo/imc-prosperity-4-backtester:/Users/you/ProsperityRepo/round3work/voucher_work/5200_work:/Users/you/ProsperityRepo/round3work/plotting/original_method/combined_analysis" \\
    python3 -c "from trader_vev5200_iv_scalp_frankfurt import Trader"

Or merge this Trader into your submission bundle.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:  # imc-prosperity-4-backtester layout
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

# --- Hardcoded session day (must match simulator / which historical CSV you calibrate to) ---
ROUND3_CSV_DAY = 0  # 0→DTE8 open, 1→7, 2→6

REPO = Path(__file__).resolve().parent.parent.parent.parent
_COMBINED = REPO / "round3work" / "plotting" / "original_method" / "combined_analysis"
if _COMBINED.is_dir():
    sys.path.insert(0, str(_COMBINED))

from plot_iv_smile_round3 import t_years_effective  # noqa: E402

from frankfurt_iv_scalp_core import (  # noqa: E402
    book_from_order_depth,
    compute_option_indicators,
    get_iv_scalping_orders_frankfurt,
    load_calibration,
    synthetic_walls_if_missing,
)

CAL_PATH = Path(__file__).resolve().parent / "calibration.json"
OPT_PRODUCT = "VEV_5200"
U_PRODUCT = "VELVETFRUIT_EXTRACT"
K_STRIKE = 5200
MAF_BID = 0
_EMA_KEY = "ema_vev5200"

# Single-file upload fallback if calibration.json is not beside this module.
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
    "TARGET_VOUCHER": "VEV_5200",
    "UNDERLYING": "VELVETFRUIT_EXTRACT",
    "WARMUP_TS_DIV100": 10,
}


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
    def bid(self) -> int:
        return int(MAF_BID)

    def run(self, state: TradingState):
        cal = _load_cal()
        sym_o = _symbol_for_product(state, OPT_PRODUCT)
        sym_u = _symbol_for_product(state, U_PRODUCT)
        store = _parse_td(getattr(state, "traderData", None))
        ema_store: dict[str, float] = store.get(_EMA_KEY) if isinstance(store.get(_EMA_KEY), dict) else {}
        if not isinstance(ema_store, dict):
            ema_store = {}
        # normalize float values
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

        T = t_years_effective(ROUND3_CSV_DAY, ts)
        ind = compute_option_indicators(cal, ema_store, u_mid, K_STRIKE, T, float(wm2), float(bb2), float(ba2), OPT_PRODUCT)

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
