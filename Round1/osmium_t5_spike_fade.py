"""
ASH_COATED_OSMIUM — t−5 spike fade with fill-based exits.

**Entry (flat only):** two-sided micro-mid vs mid from **5 observations ago**:
  - mid_now >= mid_{t-5} + 8 → open / add short (target −80).
  - mid_now <= mid_{t-5} − 8 → open / add long (target +80).

**Fills:** At each tick, position at `run()` start already includes fills from last
tick’s orders. We persist `pos_before_match` = position at the *start* of the
previous `run()`; then `fill_delta = pos_now − pos_before_match` is the net
fill since last submission. We only record when `fill_delta != 0`.

**Execution price / size:** VWAP from `state.own_trades[SYMBOL]` when the net
signed submission flow matches `fill_delta`; else fallback to the last
submitted order price for that delta.

**Exit (in position):** anchor on **entry VWAP** (updated when adds increase
|position| in the same direction). Do **not** flatten on the neutral t−5 band
while holding; only:
  - **Long:** micro-mid >= entry_vwap + 8 (shot back up from fill).
  - **Short:** micro-mid <= entry_vwap − 8 (shot back down from fill).

Also store `last_mid_t5` whenever the t−5 window is valid (for debugging / replay).

Backtest:
  PYTHONPATH="$PWD/imc-prosperity-4-backtester:$PWD/imc-prosperity-4-backtester/prosperity4bt" \\
  python3 -m prosperity4bt "$PWD/Round1/osmium_t5_spike_fade.py" 1--2 \\
  --data "$PWD/Prosperity4Data" --match-trades all --no-vis
"""
from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState, Trade

SYMBOL = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80
LOOKBACK = 5
SPIKE = 8.0
HISTORY_CAP = 100


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _vwap_from_own_trades(trades: list[Trade], expected_net: int) -> float | None:
    """
    expected_net = Δposition (buys positive, sells negative).
    Returns VWAP of the submission side for that batch, or None if inconsistent.
    """
    buy_qty = 0
    buy_notional = 0
    sell_qty = 0
    sell_notional = 0
    for t in trades:
        if t.buyer == "SUBMISSION":
            buy_qty += t.quantity
            buy_notional += t.price * t.quantity
        if t.seller == "SUBMISSION":
            sell_qty += t.quantity
            sell_notional += t.price * t.quantity
    net = buy_qty - sell_qty
    if net != expected_net:
        return None
    if expected_net > 0 and buy_qty == expected_net:
        return buy_notional / buy_qty
    if expected_net < 0 and sell_qty == -expected_net:
        return sell_notional / sell_qty
    return None


def _aggressive_toward(
    depth: OrderDepth,
    pos: int,
    target: int,
) -> list[Order]:
    diff = target - pos
    if diff > 0 and depth.sell_orders:
        best_ask = min(depth.sell_orders.keys())
        ask_vol = abs(int(depth.sell_orders[best_ask]))
        q = min(diff, POSITION_LIMIT - pos, ask_vol)
        if q > 0:
            return [Order(SYMBOL, int(best_ask), int(q))]
    if diff < 0 and depth.buy_orders:
        best_bid = max(depth.buy_orders.keys())
        bid_vol = int(depth.buy_orders[best_bid])
        q = min(-diff, POSITION_LIMIT + pos, bid_vol)
        if q > 0:
            return [Order(SYMBOL, int(best_bid), -int(q))]
    return []


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
        mid = _micro_mid(depth) if depth else None
        if mid is not None:
            history.append([float(state.timestamp), float(mid)])
            if len(history) > HISTORY_CAP:
                history = history[-HISTORY_CAP:]

        pos0 = int(state.position.get(SYMBOL, 0))
        prev_snap = store.get("pos_before_match")
        if prev_snap is not None:
            fill_delta = pos0 - int(prev_snap)
        else:
            fill_delta = 0

        entry_vwap = store.get("entry_vwap")
        entry_side = store.get("entry_side")  # +1 long, -1 short

        if fill_delta != 0:
            trades = state.own_trades.get(SYMBOL, [])
            vwap_fill = _vwap_from_own_trades(trades, fill_delta)
            if vwap_fill is None:
                lo = store.get("last_order")
                if isinstance(lo, dict) and lo.get("price") is not None:
                    vwap_fill = float(lo["price"])
                else:
                    vwap_fill = float(mid) if mid is not None else 0.0

            pos_prev = pos0 - fill_delta
            if pos_prev == 0:
                entry_side = 1 if fill_delta > 0 else -1
                entry_vwap = vwap_fill
            elif entry_side is not None and (
                (pos_prev > 0 and fill_delta > 0) or (pos_prev < 0 and fill_delta < 0)
            ):
                w0 = abs(pos_prev)
                w1 = abs(fill_delta)
                if entry_vwap is None:
                    entry_vwap = vwap_fill
                else:
                    entry_vwap = (float(entry_vwap) * w0 + float(vwap_fill) * w1) / (w0 + w1)
            elif pos0 == 0:
                entry_vwap = None
                entry_side = None
            elif (pos_prev > 0 and pos0 < 0) or (pos_prev < 0 and pos0 > 0):
                entry_vwap = None
                entry_side = None
            elif pos0 != 0 and entry_vwap is None:
                entry_side = 1 if pos0 > 0 else -1
                entry_vwap = vwap_fill

        if pos0 == 0 and fill_delta == 0:
            entry_vwap = None
            entry_side = None

        need = LOOKBACK + 1
        mid_now: float | None = None
        mid_t5: float | None = None
        if len(history) >= need:
            mid_now = float(history[-1][1])
            mid_t5 = float(history[-1 - LOOKBACK][1])
            store["last_mid_t5"] = mid_t5

        orders_out: list[Order] = []
        if (
            depth is not None
            and mid_now is not None
            and entry_vwap is not None
            and pos0 != 0
            and depth.buy_orders
            and depth.sell_orders
        ):
            ev = float(entry_vwap)
            if pos0 > 0 and mid_now >= ev + SPIKE:
                orders_out = _aggressive_toward(depth, pos0, 0)
            elif pos0 < 0 and mid_now <= ev - SPIKE:
                orders_out = _aggressive_toward(depth, pos0, 0)
        elif (
            pos0 == 0
            and depth is not None
            and mid_now is not None
            and mid_t5 is not None
            and depth.buy_orders
            and depth.sell_orders
        ):
            if mid_now >= mid_t5 + SPIKE:
                target = -POSITION_LIMIT
            elif mid_now <= mid_t5 - SPIKE:
                target = POSITION_LIMIT
            else:
                target = 0
            if target != 0:
                orders_out = _aggressive_toward(depth, pos0, target)

        if orders_out:
            result[SYMBOL] = orders_out
            o = orders_out[0]
            store["last_order"] = {"price": int(o.price), "quantity": int(o.quantity)}

        store["history"] = history
        store["pos_before_match"] = pos0
        store["entry_vwap"] = entry_vwap
        store["entry_side"] = entry_side

        return result, conversions, json.dumps(store)
