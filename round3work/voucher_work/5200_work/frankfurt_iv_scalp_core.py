"""
Frankfurt Hedgehogs IV scalping logic (single strike VEV_5200), cloned from
Prosperity3Winner/FrankfurtHedgehogs_polished.py OptionTrader:

- get_option_values: BS call + delta + vega; IV from quadratic in m_t = log(K/S)/sqrt(T)
- calculate_ema: same recurrence as ProductTrader.calculate_ema
- iv scalping: same inequalities as get_iv_scalping_orders, but uses switch_means
  (fixes polished.py reference to undefined new_switch_mean).

T and smile match round3 combined_analysis (winding DTE 8/7/6 by csv day).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

_CAL_PATH = Path(__file__).resolve().parent / "calibration.json"


def load_calibration(path: Path | None = None) -> dict[str, Any]:
    p = path or _CAL_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def _cdf(x: float) -> float:
    return float(norm.cdf(x))


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> tuple[float, float]:
    if T <= 0 or sigma <= 1e-12:
        return max(S - K, 0.0), 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    price = S * _cdf(d1) - K * math.exp(-r * T) * _cdf(d2)
    delta = _cdf(d1)
    return float(price), float(delta)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0 or sigma <= 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(S * float(norm.pdf(d1)) * math.sqrt(T))


def get_iv_smile(S: float, K: float, T: float, coeffs_high_to_low: list[float]) -> float:
    """Frankfurt get_iv: m_t_k = log(K/S)/sqrt(T); iv = poly(coeffs)(m_t_k)."""
    if S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    m_t_k = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(np.asarray(coeffs_high_to_low, dtype=float), m_t_k))


def get_option_values(
    S: float, K: float, T: float, coeffs_high_to_low: list[float]
) -> tuple[float, float, float]:
    iv = get_iv_smile(S, K, T, coeffs_high_to_low)
    if not np.isfinite(iv) or iv <= 0:
        return float("nan"), float("nan"), float("nan")
    theo, delta = bs_call(S, K, T, iv, 0.0)
    vega = bs_vega(S, K, T, iv, 0.0)
    return theo, delta, vega


def calculate_ema(ema_store: dict[str, float], key: str, window: int, value: float) -> float:
    old_mean = float(ema_store.get(key, 0.0))
    alpha = 2.0 / (window + 1.0)
    new_mean = alpha * value + (1.0 - alpha) * old_mean
    ema_store[key] = new_mean
    return new_mean


def book_from_order_depth(depth: Any) -> tuple[dict[int, int], dict[int, int], float | None, float | None, float | None, float | None, float | None]:
    """Same wall / best bid-ask semantics as ProductTrader.get_order_depth + get_walls (competition OrderDepth)."""
    buys: dict[int, int] = {}
    sells: dict[int, int] = {}
    try:
        raw_buys = getattr(depth, "buy_orders", None) or {}
        raw_sells = getattr(depth, "sell_orders", None) or {}
        buys = {int(bp): abs(int(bv)) for bp, bv in sorted(raw_buys.items(), key=lambda x: x[0], reverse=True)}
        sells = {int(sp): abs(int(sv)) for sp, sv in sorted(raw_sells.items(), key=lambda x: x[0])}
    except (TypeError, ValueError, KeyError):
        pass
    if not buys and not sells:
        return buys, sells, None, None, None, None, None
    bid_wall = min(buys.keys()) if buys else None
    ask_wall = max(sells.keys()) if sells else None
    best_bid = max(buys.keys()) if buys else None
    best_ask = min(sells.keys()) if sells else None
    wall_mid: float | None = None
    if bid_wall is not None and ask_wall is not None:
        wall_mid = (float(bid_wall) + float(ask_wall)) / 2.0
    return buys, sells, bid_wall, ask_wall, best_bid, best_ask, wall_mid


def book_from_row(
    row: dict[str, Any],
) -> tuple[dict[int, int], dict[int, int], float | None, float | None, float | None, float | None, float | None]:
    """Return (buy_orders, sell_orders, bid_wall, ask_wall, best_bid, best_ask, wall_mid)."""
    buys: dict[int, int] = {}
    sells: dict[int, int] = {}
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        ap = row.get(f"ask_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        av = row.get(f"ask_volume_{i}")
        if bp is not None and bv is not None and pd_notna(bv) and int(bv) > 0:
            buys[int(bp)] = abs(int(bv))
        if ap is not None and av is not None and pd_notna(av) and int(av) > 0:
            sells[int(ap)] = abs(int(av))
    if not buys and not sells:
        return buys, sells, None, None, None, None, None
    bid_wall = min(buys.keys()) if buys else None
    ask_wall = max(sells.keys()) if sells else None
    best_bid = max(buys.keys()) if buys else None
    best_ask = min(sells.keys()) if sells else None
    wall_mid: float | None = None
    if bid_wall is not None and ask_wall is not None:
        wall_mid = (float(bid_wall) + float(ask_wall)) / 2.0
    return buys, sells, bid_wall, ask_wall, best_bid, best_ask, wall_mid


def pd_notna(x: Any) -> bool:
    try:
        import pandas as pd

        return bool(pd.notna(x))
    except Exception:
        return x is not None and not (isinstance(x, float) and math.isnan(x))


def synthetic_walls_if_missing(
    bid_wall: float | None,
    ask_wall: float | None,
    best_bid: float | None,
    best_ask: float | None,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Frankfurt fallbacks; returns (bid_wall, ask_wall, wall_mid, best_bid, best_ask)."""
    wall_mid = None
    if bid_wall is None and ask_wall is not None:
        wall_mid = float(ask_wall) - 0.5
        bid_wall = float(ask_wall) - 1.0
        best_bid = float(ask_wall) - 1.0
    elif ask_wall is None and bid_wall is not None:
        wall_mid = float(bid_wall) + 0.5
        ask_wall = float(bid_wall) + 1.0
        best_ask = float(bid_wall) + 1.0
    elif bid_wall is not None and ask_wall is not None:
        wall_mid = (bid_wall + ask_wall) / 2.0
    return bid_wall, ask_wall, wall_mid, best_bid, best_ask


def compute_option_indicators(
    cal: dict[str, Any],
    ema_store: dict[str, float],
    underlying_mid: float,
    K: int,
    T: float,
    wall_mid: float | None,
    best_bid: float | None,
    best_ask: float | None,
    option_name: str,
) -> dict[str, Any]:
    """Single-option slice of OptionTrader.calculate_indicators (underlying branch omitted)."""
    out: dict[str, Any] = {
        "current_theo_diff": None,
        "mean_theo_diff": None,
        "switch_mean": None,
        "vega": None,
        "delta": None,
    }
    coeffs = cal["coeffs_high_to_low"]
    if wall_mid is None or best_bid is None or best_ask is None:
        return out
    theo, delta, vega = get_option_values(float(underlying_mid), float(K), float(T), coeffs)
    if not np.isfinite(theo):
        return out
    option_theo_diff = float(wall_mid) - theo
    out["current_theo_diff"] = option_theo_diff
    out["vega"] = vega
    out["delta"] = delta

    wn = int(cal["THEO_NORM_WINDOW"])
    sw = int(cal["IV_SCALPING_WINDOW"])
    new_mean_diff = calculate_ema(ema_store, f"{option_name}_theo_diff", wn, option_theo_diff)
    out["mean_theo_diff"] = new_mean_diff
    new_mean_avg_dev = calculate_ema(
        ema_store, f"{option_name}_avg_devs", sw, abs(option_theo_diff - new_mean_diff)
    )
    out["switch_mean"] = new_mean_avg_dev
    return out


def get_iv_scalping_orders_frankfurt(
    cal: dict[str, Any],
    ind: dict[str, Any],
    wall_mid: float,
    best_bid: float,
    best_ask: float,
    initial_position: int,
    max_allowed_buy_volume: int,
    max_allowed_sell_volume: int,
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """
    Returns (bid_orders, ask_orders) as list of (price, qty) for buys and (price, qty) for sells
    where qty is positive for buys and positive for sells (we'll negate for ask semantics).

    Mirrors get_iv_scalping_orders; uses switch_means key passed as ind['switch_mean'].
    """
    bids: list[tuple[int, int]] = []
    asks: list[tuple[int, int]] = []

    mean_theo_diff = ind.get("mean_theo_diff")
    current_theo_diff = ind.get("current_theo_diff")
    switch_mean = ind.get("switch_mean")
    vega = float(ind.get("vega") or 0.0)

    if mean_theo_diff is None or current_theo_diff is None or switch_mean is None:
        return bids, asks

    THR_OPEN = float(cal["THR_OPEN"])
    THR_CLOSE = float(cal["THR_CLOSE"])
    LOW_VEGA_THR_ADJ = float(cal["LOW_VEGA_THR_ADJ"])
    IV_SCALPING_THR = float(cal["IV_SCALPING_THR"])
    low_vega_cut = float(cal["LOW_VEGA_CUTOFF"])

    low_vega_adj = 0.0
    if vega <= low_vega_cut:
        low_vega_adj = LOW_VEGA_THR_ADJ

    # --- polished uses self.new_switch_mean; we use switch_mean from indicators ---
    if switch_mean >= IV_SCALPING_THR:
        if (
            current_theo_diff - wall_mid + best_bid - mean_theo_diff >= (THR_OPEN + low_vega_adj)
            and max_allowed_sell_volume > 0
        ):
            asks.append((int(best_bid), int(max_allowed_sell_volume)))
        if (
            current_theo_diff - wall_mid + best_bid - mean_theo_diff >= THR_CLOSE
            and initial_position > 0
        ):
            asks.append((int(best_bid), int(initial_position)))
        elif (
            current_theo_diff - wall_mid + best_ask - mean_theo_diff <= -(THR_OPEN + low_vega_adj)
            and max_allowed_buy_volume > 0
        ):
            bids.append((int(best_ask), int(max_allowed_buy_volume)))
        if (
            current_theo_diff - wall_mid + best_ask - mean_theo_diff <= -THR_CLOSE
            and initial_position < 0
        ):
            bids.append((int(best_ask), int(-initial_position)))
    else:
        if initial_position > 0:
            asks.append((int(best_bid), int(initial_position)))
        elif initial_position < 0:
            bids.append((int(best_ask), int(-initial_position)))

    return bids, asks
