"""
INTARIAN_PEPPER_ROOT — rolling mid slope over the last WINDOW observations.

1. No trades until WINDOW valid mids are stored (each `run` with a two-sided book
   appends [timestamp, mid] to history in traderData).
2. Once len(history) >= WINDOW, OLS slope of mid vs timestamp on that window.
   slope > 0 -> target +80; slope < 0 -> -80; slope == 0 -> 0 flat.
3. Each tick, send aggressive orders toward target (best ask buys, best bid sells),
   clipped by position limit and top-of-book size.

traderData JSON: {"history": [[ts, mid], ...]}  (at most WINDOW entries).

Backtest (from ProsperityRepo — PYTHONPATH required or `datamodel` import fails):
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/pepper_rolling_slope_trader.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades all --no-vis
"""
from __future__ import annotations

import json
from datamodel import Order, TradingState

SYMBOL = "INTARIAN_PEPPER_ROOT"
POSITION_LIMIT = 80
WINDOW = 50


def _mid_from_depth(depth) -> float | None:
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None
    best_bid = max(depth.buy_orders.keys())
    best_ask = min(depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0


def _ols_slope_mid_vs_time(points: list[list[float]]) -> float:
    """Least-squares slope y ~ a*t + b. points: [[t, y], ...]."""
    n = len(points)
    if n < 2:
        return 0.0
    sum_t = 0.0
    sum_y = 0.0
    sum_tt = 0.0
    sum_ty = 0.0
    for t, y in points:
        sum_t += t
        sum_y += y
        sum_tt += t * t
        sum_ty += t * y
    den = n * sum_tt - sum_t * sum_t
    if den == 0.0:
        return 0.0
    return (n * sum_ty - sum_t * sum_y) / den


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        history: list[list[float]] = store.get("history", [])

        depth = state.order_depths.get(SYMBOL)
        mid = _mid_from_depth(depth)
        if mid is not None:
            history.append([float(state.timestamp), float(mid)])
            if len(history) > WINDOW:
                history = history[-WINDOW:]

        pos = int(state.position.get(SYMBOL, 0))

        if len(history) < WINDOW:
            store["history"] = history
            return result, conversions, json.dumps(store)

        window = history[-WINDOW:]
        slope = _ols_slope_mid_vs_time(window)

        if slope > 0.0:
            target = POSITION_LIMIT
        elif slope < 0.0:
            target = -POSITION_LIMIT
        else:
            target = 0

        diff = target - pos
        if diff != 0 and depth is not None:
            if diff > 0 and depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                ask_vol = abs(int(depth.sell_orders[best_ask]))
                q = min(diff, POSITION_LIMIT - pos, ask_vol)
                if q > 0:
                    result[SYMBOL] = [Order(SYMBOL, int(best_ask), int(q))]
            elif diff < 0 and depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                bid_vol = int(depth.buy_orders[best_bid])
                q = min(-diff, POSITION_LIMIT + pos, bid_vol)
                if q > 0:
                    result[SYMBOL] = [Order(SYMBOL, int(best_bid), -int(q))]

        store["history"] = history
        return result, conversions, json.dumps(store)
