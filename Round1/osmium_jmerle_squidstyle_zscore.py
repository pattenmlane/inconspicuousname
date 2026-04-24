"""
ASH_COATED_OSMIUM — jmerle-style Prosperity 3 squid ink adapted for Prosperity 4.

From INK_INFO/jmerle.py SquidInkStrategy:
  - Mid = (most-volume bid price + most-volume ask price) / 2 (not simple micro-mid).
  - Append mid each tick; keep last (zscore_period + smoothing_period) = 250 values.
  - score = rolling_mean_100( (mid - rolling_mean_150(mid)) / rolling_std_150(mid) )
  - score < -1  -> LONG  (fade low stretch)  -> aggressive buys toward +limit
  - score > +1  -> SHORT (fade high stretch) -> aggressive sells toward -limit
  - else -> keep prior signal; NEUTRAL flattens to 0.

Position limit 80 matches imc-prosperity-4-backtester constants for this symbol.

Backtest (repo root, PYTHONPATH):
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/osmium_jmerle_squidstyle_zscore.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades all --no-vis

Optional env (override jmerle defaults for snappier z on osmium):
  OSMIUM_JMERLE_WZ     z-score window (default 150)
  OSMIUM_JMERLE_WS     smoothing window on z (default 100)
  OSMIUM_JMERLE_THRESH absolute score threshold (default 1.0)

Example:
  OSMIUM_JMERLE_WZ=20 OSMIUM_JMERLE_WS=15 python3 -m prosperity4bt ...
"""
from __future__ import annotations

import json
import os
import statistics
from datamodel import Order, OrderDepth, TradingState

SYMBOL = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(str(raw).strip())


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return float(str(raw).strip())


ZSCORE_PERIOD = max(2, _int_env("OSMIUM_JMERLE_WZ", 150))
SMOOTHING_PERIOD = max(1, _int_env("OSMIUM_JMERLE_WS", 100))
THRESHOLD = _float_env("OSMIUM_JMERLE_THRESH", 1.0)
REQUIRED = ZSCORE_PERIOD + SMOOTHING_PERIOD

# Signal values (match jmerle Signal enum)
NEUTRAL = 0
SHORT = 1
LONG = 2


def _mid_popular_volume(depth: OrderDepth) -> float | None:
    """jmerle get_mid_price: average of largest-size bid and smallest-size ask levels."""
    if not depth.buy_orders or not depth.sell_orders:
        return None
    buy_orders = sorted(depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(depth.sell_orders.items())
    popular_buy = max(buy_orders, key=lambda t: t[1])[0]
    popular_sell = min(sell_orders, key=lambda t: t[1])[0]
    return (popular_buy + popular_sell) / 2.0


def _rolling_smoothed_z_tail(prices: list[float]) -> float | None:
    """
    Last value of: rolling_mean(SMOOTHING) of rolling_z(ZSCORE) on prices.
    Aligns with pandas: rolling(window).mean() / .std() (sample std, ddof=1).
    """
    n = len(prices)
    if n < REQUIRED:
        return None

    z_tail: list[float] = []
    for end in range(ZSCORE_PERIOD - 1, n):
        w = prices[end - ZSCORE_PERIOD + 1 : end + 1]
        m = statistics.mean(w)
        if len(w) < 2:
            z_tail.append(0.0)
            continue
        try:
            s = statistics.stdev(w)
        except statistics.StatisticsError:
            s = 0.0
        if s <= 0.0:
            z_tail.append(0.0)
        else:
            z_tail.append((prices[end] - m) / s)

    if len(z_tail) < SMOOTHING_PERIOD:
        return None
    smooth_window = z_tail[-SMOOTHING_PERIOD:]
    return float(statistics.mean(smooth_window))


def _next_signal(score: float | None) -> int | None:
    if score is None:
        return None
    if score < -THRESHOLD:
        return LONG
    if score > THRESHOLD:
        return SHORT
    return None


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        signal = int(store.get("signal", NEUTRAL))
        history: list[float] = store.get("history", [])

        depth = state.order_depths.get(SYMBOL)
        mid = _mid_popular_volume(depth) if depth else None
        if mid is not None:
            history.append(float(mid))
            while len(history) > REQUIRED:
                history.pop(0)

        new_sig = _next_signal(_rolling_smoothed_z_tail(history))
        if new_sig is not None:
            signal = new_sig

        pos = int(state.position.get(SYMBOL, 0))

        if depth is None or not depth.buy_orders or not depth.sell_orders:
            store["signal"] = signal
            store["history"] = history
            return result, conversions, json.dumps(store)

        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())
        ask_vol = abs(int(depth.sell_orders[best_ask]))
        bid_vol = int(depth.buy_orders[best_bid])

        if signal == NEUTRAL:
            if pos < 0:
                q = min(-pos, ask_vol)
                if q > 0:
                    result[SYMBOL] = [Order(SYMBOL, int(best_ask), int(q))]
            elif pos > 0:
                q = min(pos, bid_vol)
                if q > 0:
                    result[SYMBOL] = [Order(SYMBOL, int(best_bid), -int(q))]
        elif signal == SHORT:
            q = min(POSITION_LIMIT + pos, bid_vol)
            if q > 0:
                result[SYMBOL] = [Order(SYMBOL, int(best_bid), -int(q))]
        elif signal == LONG:
            q = min(POSITION_LIMIT - pos, ask_vol)
            if q > 0:
                result[SYMBOL] = [Order(SYMBOL, int(best_ask), int(q))]

        store["signal"] = signal
        store["history"] = history
        return result, conversions, json.dumps(store)
