"""
ASH_COATED_OSMIUM — isolated Alpha Animals *squid ink* style logic (Prosperity 3),
ported to Prosperity 4 backtester datamodel.

Idea (from ``INK_INFO/alphaanimals.py`` ``squid_ink_strategy``):
  - Track micro-mid history; rolling mean over ``MEAN_WINDOW``; short-horizon volatility.
  - When vol and %-deviation from mean exceed thresholds, **fade the stretch**:
      mid > mean → short at best bid; mid < mean → long at best ask
    (Original also gated on insider regime; P4 has no Olivia here — regime checks
    are disabled by keeping ``regime`` always ``None``, so only mean / vol logic applies.)
  - Manage open leg: exit on mean reversion or max hold (wall-clock ``timestamp`` delta).

**Volatility / monetization:** edge scales with recent stdev of mids (wider moves →
slightly easier trigger on ``deviation_pct``). Tune constants below.

**Desktop / paths:** Moving the repo is fine — open that folder in Cursor and run
backtests with ``--data`` pointing at your ``Prosperity4Data`` (any path is OK).

Backtest:
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/osmium_alphaanimals_ink_style.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades worse --no-vis
"""
from __future__ import annotations

import json
import statistics
from datamodel import Order, OrderDepth, TradingState

SYMBOL = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80

# From alpha animals (ink) — tuned names for osmium
TIMESPAN = 20
MEAN_WINDOW = 30
VOL_WINDOW = 10
VOLATILITY_THRESHOLD = 3.0
DEVIATION_THRESHOLD = 0.05
# Original used raw timestamp diff; P4 timestamps often step by ~100 → use absolute time
MAX_HOLD_TIMESTAMP_DELTA = 50_000


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


class Trader:
    def __init__(self) -> None:
        self._prices: list[float] = []
        self._position_start_ts: int = 0

    def _vol_scaled_deviation_threshold(self) -> float:
        if len(self._prices) < 2:
            return DEVIATION_THRESHOLD
        w = self._prices[-min(VOL_WINDOW, len(self._prices)) :]
        vol = statistics.stdev(w) if len(w) >= 2 else 0.0
        # Higher vol → slightly lower bar to trade (monetize vol); clamp sensibly
        scale = max(0.75, min(1.5, 1.0 + (vol - VOLATILITY_THRESHOLD) * 0.05))
        return DEVIATION_THRESHOLD / scale

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        self._prices = store.get("prices", [])
        self._position_start_ts = int(store.get("position_start_ts", 0))

        depth = state.order_depths.get(SYMBOL)
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            store["prices"] = self._prices
            store["position_start_ts"] = self._position_start_ts
            return result, conversions, json.dumps(store)

        mid = _micro_mid(depth)
        if mid is None:
            store["prices"] = self._prices
            store["position_start_ts"] = self._position_start_ts
            return result, conversions, json.dumps(store)

        self._prices.append(float(mid))
        cap = max(TIMESPAN, MEAN_WINDOW) + 5
        if len(self._prices) > cap:
            self._prices = self._prices[-cap:]

        position = int(state.position.get(SYMBOL, 0))
        ts = int(state.timestamp)
        regime = None  # no insider on P4 tape; keep ink mean-reversion only

        best_ask = min(depth.sell_orders.keys())
        best_bid = max(depth.buy_orders.keys())

        orders: list[Order] = []

        if len(self._prices) < 10:
            store["prices"] = self._prices
            store["position_start_ts"] = self._position_start_ts
            return result, conversions, json.dumps(store)

        recent_window = min(len(self._prices), MEAN_WINDOW)
        mean_price = sum(self._prices[-recent_window:]) / recent_window

        w = self._prices[-min(VOL_WINDOW, len(self._prices)) :]
        volatility = statistics.stdev(w) if len(w) >= 2 else 0.0

        deviation_pct = abs(mid - mean_price) / mean_price if mean_price > 0 else 0.0
        dev_thr = self._vol_scaled_deviation_threshold()

        # Force-close stale positions (timestamp-based, P4-safe)
        if position != 0 and self._position_start_ts > 0:
            if ts - self._position_start_ts >= MAX_HOLD_TIMESTAMP_DELTA:
                orders.extend(self._close_all(depth, position))
                if orders:
                    result[SYMBOL] = orders
                self._position_start_ts = 0
                store["prices"] = self._prices
                store["position_start_ts"] = self._position_start_ts
                return result, conversions, json.dumps(store)

        if position == 0:
            if volatility > VOLATILITY_THRESHOLD and deviation_pct > dev_thr:
                if mid > mean_price and regime != "bullish":
                    q = min(depth.buy_orders[best_bid], POSITION_LIMIT)
                    if q > 0:
                        orders.append(Order(SYMBOL, int(best_bid), -int(q)))
                        self._position_start_ts = ts
                elif mid < mean_price and regime != "bearish":
                    q = min(-depth.sell_orders[best_ask], POSITION_LIMIT)
                    if q > 0:
                        orders.append(Order(SYMBOL, int(best_ask), int(q)))
                        self._position_start_ts = ts
        else:
            if position > 0:
                if mid >= mean_price or regime == "bearish":
                    q = min(position, depth.buy_orders[best_bid])
                    if q > 0:
                        orders.append(Order(SYMBOL, int(best_bid), -int(q)))
                        if q >= position:
                            self._position_start_ts = 0
            elif position < 0:
                if mid <= mean_price or regime == "bullish":
                    q = min(-position, -depth.sell_orders[best_ask])
                    if q > 0:
                        orders.append(Order(SYMBOL, int(best_ask), int(q)))
                        if q >= -position:
                            self._position_start_ts = 0

        if orders:
            result[SYMBOL] = orders

        store["prices"] = self._prices
        store["position_start_ts"] = self._position_start_ts
        return result, conversions, json.dumps(store)

    def _close_all(self, depth: OrderDepth, position: int) -> list[Order]:
        out: list[Order] = []
        if position > 0 and depth.buy_orders:
            b = max(depth.buy_orders.keys())
            q = min(position, depth.buy_orders[b])
            if q > 0:
                out.append(Order(SYMBOL, int(b), -int(q)))
        elif position < 0 and depth.sell_orders:
            a = min(depth.sell_orders.keys())
            q = min(-position, -depth.sell_orders[a])
            if q > 0:
                out.append(Order(SYMBOL, int(a), int(q)))
        return out
