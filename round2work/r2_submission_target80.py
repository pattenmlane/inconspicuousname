from __future__ import annotations

"""Round 2 working submission (Trader module for IMC Prosperity backtester).

Pepper (INTARIAN_PEPPER_ROOT): Drift-anchored emerald market making. Fair
value is ``alpha + BETA_DRIFT * timestamp``, with ``alpha`` set once from the
first touch mid so FV drifts slowly upward. Target long **70** of **80** max
position: lift asks below FV (size capped by room / need toward target), hit
bids above FV, passive bid at ``bbbf+1`` and ask at ``baaf-1``, plus inventory
clears at floor/ceil FV when those prices rest on the book. **Slope safeguard:**
keep the last ``PEPPER_SLOPE_WINDOW`` touch mids; if OLS slope of mid vs time
is below ``PEPPER_SLOPE_SAFEGUARD``, skip drift MM for that tick and send one
aggressive order pushing toward **-position_limit** (crash / breakdown mode).

Osmium (ASH_COATED_OSMIUM): Fair value from **wall mid** (min bid price and
max ask price on the book, averaged), else touch mid, else 10_000. Emerald width
**2**: passive bid/ask around FV, take when best ask < FV or best bid > FV,
inventory clears at floor/ceil FV. **WM spike freeze:** if
``|wall_mid − previous_wall_mid| ≥ OSMIUM_WM_SPIKE`` (in price ticks), freeze
fair value at the last good level for ``OSMIUM_WM_FREEZE_TICKS`` ticks
(state: ``osm_prev_wall``, ``osm_last_fv``, ``osm_freeze_left``,
``osm_frozen_fv`` in ``traderData`` JSON). Structured osmium logs print as
``OSMIUM_WMFN_JSON`` + JSON lines.

**Market Access Fee (Round 2 only):** ``Trader.bid()`` returns how many
**XIRECs** you offer for extra order-book flow. The organizer ranks all
submissions; roughly the **top 50%** of bids win access and pay their bid
(off **final** Round 2 profit). Local backtests ignore ``bid()``; it only
matters in the final R2 simulation.
"""

# Only edit vs ``r2_submission.py``: ``PEPPER_TARGET_LONG`` below is **80** (original is 70).

import json
import math
from typing import List

from datamodel import Order, OrderDepth, TradingState

BETA_DRIFT = 1.0e-3
PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"
WIDTH = 2
OSMIUM_POSITION_LIMIT = 80

FALLBACK_OSMIUM = 10_000.0
LOG_OSMIUM = "OSMIUM_WMFN_JSON"

# Pepper 70/10 (edit here instead of env vars)
PEPPER_TARGET_LONG = 80
PEPPER_POSITION_LIMIT = 80

# Rolling touch-mid slope safeguard (dy/dt vs timestamp). If OLS slope on the last
# ``PEPPER_SLOPE_WINDOW`` mids falls below ``PEPPER_SLOPE_SAFEGUARD``, skip drift
# MM and aggressively push inventory toward **-PEPPER_POSITION_LIMIT** (short).
PEPPER_SLOPE_WINDOW = 50
PEPPER_SLOPE_SAFEGUARD = -0.00015
PEPPER_SLOPE_HIST = "pepper_slope_hist"

# Osmium spike freeze (edit here instead of OSMIUM_WM_* env)
OSMIUM_WM_SPIKE = 3.0
OSMIUM_WM_FREEZE_TICKS = 5

# Round 2 Market Access Fee (MAF): XIRECs bid for ~top-half extra book flow (blind auction).
MAF_BID_XIREC = 6001


def _target_long() -> int:
    lim = _position_limit()
    return max(0, min(PEPPER_TARGET_LONG, lim))


def _position_limit() -> int:
    return max(1, PEPPER_POSITION_LIMIT)


def _spike_threshold() -> float:
    return max(0.0, float(OSMIUM_WM_SPIKE))


def _freeze_ticks() -> int:
    return max(0, int(OSMIUM_WM_FREEZE_TICKS))


def _store_float(x: object) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def _store_nonneg_int(x: object, default: int = 0) -> int:
    if isinstance(x, bool):
        return default
    if isinstance(x, int):
        return max(0, x)
    if isinstance(x, float) and math.isfinite(x):
        return max(0, int(x))
    if isinstance(x, str):
        try:
            v = float(x)
            if math.isfinite(v):
                return max(0, int(v))
        except ValueError:
            pass
    return default


def _book_volume_int(v: object) -> int:
    """Order book sizes must be ints; coerce defensively for live / odd tapes."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _parse_trader_data(raw: object) -> dict:
    """Parse persisted JSON; never raise — bad or missing data starts fresh."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except Exception:
            return {}
    s = str(raw).strip() if raw else ""
    if not s:
        return {}
    try:
        out = json.loads(s)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _position_int(state: TradingState, sym: str) -> int:
    try:
        v = state.position.get(sym, 0)
        return int(v)
    except (TypeError, ValueError):
        return 0


def _timestamp_int(state: TradingState) -> int:
    try:
        return int(state.timestamp)
    except (TypeError, ValueError, AttributeError):
        return 0


def _finite_or(x: float, fallback: float) -> float:
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return fallback


def _micro_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


def _sanitize_pepper_slope_hist(raw: object, window: int) -> list[list[float]]:
    if not isinstance(raw, list) or window <= 0:
        return []
    out: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            t = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(y):
            out.append([t, y])
    return out[-window * 2 :]


def _ols_slope_mid_vs_time(points: list[list[float]]) -> float:
    n = len(points)
    if n < 2:
        return 0.0
    sum_t = sum_y = sum_tt = sum_ty = 0.0
    for t, y in points:
        sum_t += t
        sum_y += y
        sum_tt += t * t
        sum_ty += t * y
    den = n * sum_tt - sum_t * sum_t
    if den == 0.0:
        return 0.0
    return (n * sum_ty - sum_t * sum_y) / den


def _wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid_wall = min(depth.buy_orders.keys())
    ask_wall = max(depth.sell_orders.keys())
    return (bid_wall + ask_wall) / 2.0


def _fair_osmium_no_store(depth: OrderDepth) -> float:
    w = _wall_mid(depth)
    if w is not None:
        return float(w)
    m = _micro_mid(depth)
    if m is not None:
        return float(m)
    return FALLBACK_OSMIUM


class Trader:
    def _log_osm(self, obj: dict) -> None:
        try:
            print(LOG_OSMIUM + json.dumps(obj, separators=(",", ":")))
        except Exception:
            pass

    def bid(self) -> int:
        return MAF_BID_XIREC

    # ----- Pepper (70/10 drift MM) -----
    def pepper_emerald_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
        target_long: int,
    ) -> List[Order]:
        orders: List[Order] = []

        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2

        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * _book_volume_int(order_depth.sell_orders[best_ask])
            cap_room = position_limit - position
            need = max(0, target_long - position)
            if best_ask < fair_value:
                quantity = min(best_ask_amount, cap_room)
            else:
                quantity = min(best_ask_amount, cap_room, need) if need > 0 else 0
            if quantity > 0:
                orders.append(Order(PEPPER, int(best_ask), quantity))

        buy_vol_takes = sum(o.quantity for o in orders if o.quantity > 0)
        sell_vol_takes = sum(-o.quantity for o in orders if o.quantity < 0)
        pos_after_buys = position + buy_vol_takes

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = int(order_depth.buy_orders[best_bid])
            if best_bid > fair_value:
                max_sell = max(0, pos_after_buys - target_long)
                quantity = min(best_bid_amount, position_limit + position, max_sell)
                if quantity > 0:
                    orders.append(Order(PEPPER, int(best_bid), -quantity))

        buy_order_volume, sell_order_volume = self.pepper_clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            PEPPER,
            sum([o.quantity for o in orders if o.quantity > 0]),
            sum([-o.quantity for o in orders if o.quantity < 0]),
            fair_value,
            width,
            target_long,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(PEPPER, int(bbbf + 1), buy_quantity))

        pos_after = position + buy_order_volume - sell_order_volume
        max_passive_sell = max(0, pos_after - target_long)
        std_sell = position_limit + (position - sell_order_volume)
        sell_quantity = min(std_sell, max_passive_sell)
        if sell_quantity > 0:
            orders.append(Order(PEPPER, int(baaf - 1), -sell_quantity))

        return orders

    def pepper_clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int,
        target_long: int,
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            cap = max(0, position_after_take - target_long)
            sent_quantity = min(sell_quantity, clear_quantity, cap)
            if sent_quantity > 0:
                orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(_book_volume_int(order_depth.sell_orders[fair_for_bid])), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def _pepper_crash_reversal_orders(
        self, order_depth: OrderDepth, position: int, position_limit: int
    ) -> List[Order]:
        """Aggressive inventory toward max short when rolling mid slope looks broken."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []
        target = -position_limit
        diff = target - position
        if diff < 0:
            best_bid = max(order_depth.buy_orders.keys())
            bid_vol = _book_volume_int(order_depth.buy_orders[best_bid])
            q = min(-diff, position_limit + position, bid_vol)
            if q > 0:
                return [Order(PEPPER, int(best_bid), -int(q))]
        if diff > 0:
            best_ask = min(order_depth.sell_orders.keys())
            ask_vol = abs(_book_volume_int(order_depth.sell_orders[best_ask]))
            q = min(diff, position_limit - position, ask_vol)
            if q > 0:
                return [Order(PEPPER, int(best_ask), int(q))]
        return []

    # ----- Osmium (wall mid + spike freeze) -----
    def osmium_emerald_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        width: int,
        position: int,
        position_limit: int,
        ts: int,
        wall_raw: float | None,
        freeze_active: bool,
        freeze_left_start: int,
        freeze_left_end: int,
        prev_wall: float | None,
        spike_this_tick: bool,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        sell_above_fv = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        baaf = min(sell_above_fv) if sell_above_fv else fair_value + 2

        buy_below_fv = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        bbbf = max(buy_below_fv) if buy_below_fv else fair_value - 2

        self._log_osm(
            {
                "t": ts,
                "event": "tick_context",
                "fv": fair_value,
                "wall_raw": wall_raw,
                "prev_wall": prev_wall,
                "freeze_active": freeze_active,
                "freeze_left_start": freeze_left_start,
                "freeze_left_end": freeze_left_end,
                "spike_this_tick": spike_this_tick,
                "pos": position,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bbbf": float(bbbf),
                "baaf": float(baaf),
            }
        )

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    self._log_osm(
                        {
                            "t": ts,
                            "event": "intent",
                            "kind": "take_buy_best_ask_below_fv",
                            "human": f"best_ask={best_ask} < fv={fair_value} → lift ask (buy) qty={quantity}",
                            "price": int(best_ask),
                            "qty": int(quantity),
                        }
                    )
                    orders.append(Order(OSMIUM, int(best_ask), quantity))

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = _book_volume_int(order_depth.buy_orders[best_bid])
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    self._log_osm(
                        {
                            "t": ts,
                            "event": "intent",
                            "kind": "take_sell_best_bid_above_fv",
                            "human": f"best_bid={best_bid} > fv={fair_value} → hit bid (sell) qty={quantity}",
                            "price": int(best_bid),
                            "qty": int(quantity),
                        }
                    )
                    orders.append(Order(OSMIUM, int(best_bid), -quantity))

        buy_order_volume, sell_order_volume = self.osmium_clear_position_order(
            orders,
            order_depth,
            position,
            position_limit,
            OSMIUM,
            sum([o.quantity for o in orders if o.quantity > 0]),
            sum([-o.quantity for o in orders if o.quantity < 0]),
            fair_value,
            width,
            ts,
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            px = int(bbbf + 1)
            self._log_osm(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "passive_mm_buy",
                    "human": f"passive bid (mm) price={px} qty={buy_quantity} (bbbf+1, bbbf={bbbf})",
                    "price": px,
                    "qty": int(buy_quantity),
                }
            )
            orders.append(Order(OSMIUM, px, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            px = int(baaf - 1)
            self._log_osm(
                {
                    "t": ts,
                    "event": "intent",
                    "kind": "passive_mm_sell",
                    "human": f"passive ask (mm) price={px} qty={sell_quantity} (baaf-1, baaf={baaf})",
                    "price": px,
                    "qty": int(sell_quantity),
                }
            )
            orders.append(Order(OSMIUM, px, -sell_quantity))

        return orders

    def osmium_clear_position_order(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int,
        ts: int,
    ):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                self._log_osm(
                    {
                        "t": ts,
                        "event": "intent",
                        "kind": "clear_long_sell_at_ceil_fv",
                        "human": f"clear long: sell {sent_quantity} @ ceil(fv)={fair_for_ask}",
                        "price": int(fair_for_ask),
                        "qty": int(sent_quantity),
                    }
                )
                orders.append(Order(product, int(fair_for_ask), -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                self._log_osm(
                    {
                        "t": ts,
                        "event": "intent",
                        "kind": "clear_short_buy_at_floor_fv",
                        "human": f"clear short: buy {sent_quantity} @ floor(fv)={fair_for_bid}",
                        "price": int(fair_for_bid),
                        "qty": int(sent_quantity),
                    }
                )
                orders.append(Order(product, int(fair_for_bid), abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        store = _parse_trader_data(getattr(state, "traderData", None))

        depths = getattr(state, "order_depths", None)
        if not isinstance(depths, dict):
            depths = {}

        # --- Pepper (drift MM + rolling slope crash safeguard) ---
        target_long = _target_long()
        position_limit_pe = _position_limit()
        alpha = _store_float(store.get("alpha"))
        depth_pe = depths.get(PEPPER)
        mid_pe = _micro_mid(depth_pe) if depth_pe else None

        hist = _sanitize_pepper_slope_hist(store.get(PEPPER_SLOPE_HIST), PEPPER_SLOPE_WINDOW)
        ts = _timestamp_int(state)
        if mid_pe is not None:
            hist.append([float(ts), float(mid_pe)])
            if len(hist) > PEPPER_SLOPE_WINDOW * 2:
                hist = hist[-(PEPPER_SLOPE_WINDOW * 2) :]
        store[PEPPER_SLOPE_HIST] = hist

        slope_crash = False
        if (
            len(hist) >= PEPPER_SLOPE_WINDOW
            and depth_pe is not None
            and depth_pe.buy_orders
            and depth_pe.sell_orders
        ):
            slope = _ols_slope_mid_vs_time(hist[-PEPPER_SLOPE_WINDOW:])
            slope_crash = slope < PEPPER_SLOPE_SAFEGUARD

        if slope_crash and PEPPER in depths:
            pos_pe = _position_int(state, PEPPER)
            result[PEPPER] = self._pepper_crash_reversal_orders(
                depths[PEPPER], pos_pe, position_limit_pe
            )
        else:
            if alpha is None and mid_pe is not None:
                alpha = float(mid_pe) - BETA_DRIFT * float(ts)
                store["alpha"] = alpha

            if alpha is not None and PEPPER in depths:
                drifted = float(alpha) + BETA_DRIFT * float(ts)
                fair_pe = _finite_or(drifted, float(alpha))
                pos_pe = _position_int(state, PEPPER)
                result[PEPPER] = self.pepper_emerald_orders(
                    depths[PEPPER],
                    fair_pe,
                    WIDTH,
                    pos_pe,
                    position_limit_pe,
                    target_long,
                )

        # --- Osmium spike freeze ---
        if OSMIUM in depths:
            depth_os = depths[OSMIUM]
            wall_raw = _wall_mid(depth_os)
            spike_thr = _spike_threshold()
            n_freeze = _freeze_ticks()

            prev_wall = _store_float(store.get("osm_prev_wall"))
            last_fv = _store_float(store.get("osm_last_fv"))
            freeze_left = _store_nonneg_int(store.get("osm_freeze_left"), 0)
            freeze_left_start = freeze_left
            frozen_fv = _store_float(store.get("osm_frozen_fv"))

            spike_this_tick = (
                wall_raw is not None
                and prev_wall is not None
                and abs(float(wall_raw) - float(prev_wall)) >= spike_thr
            )

            if spike_this_tick and n_freeze > 0:
                lf = last_fv
                fv = lf if lf is not None else float(wall_raw)
                frozen_fv = fv
                freeze_left = n_freeze - 1
            elif freeze_left > 0 and frozen_fv is not None:
                fv = frozen_fv
                freeze_left -= 1
            elif wall_raw is not None:
                fv = float(wall_raw)
                freeze_left = 0
                frozen_fv = None
            else:
                lf = last_fv
                fv = lf if lf is not None else _fair_osmium_no_store(depth_os)
                if freeze_left <= 0:
                    frozen_fv = None

            if wall_raw is not None:
                store["osm_prev_wall"] = _finite_or(float(wall_raw), FALLBACK_OSMIUM)
            store["osm_last_fv"] = _finite_or(float(fv), FALLBACK_OSMIUM)
            store["osm_freeze_left"] = freeze_left
            if frozen_fv is not None:
                store["osm_frozen_fv"] = float(frozen_fv)
            else:
                store.pop("osm_frozen_fv", None)

            freeze_active = spike_this_tick or (freeze_left_start > 0)
            fair_value = _finite_or(float(fv), FALLBACK_OSMIUM)
            pos_os = _position_int(state, OSMIUM)
            result[OSMIUM] = self.osmium_emerald_orders(
                depth_os,
                fair_value,
                WIDTH,
                pos_os,
                OSMIUM_POSITION_LIMIT,
                ts,
                wall_raw,
                freeze_active,
                freeze_left_start,
                freeze_left,
                prev_wall,
                spike_this_tick,
            )

        try:
            td_out = json.dumps(store, separators=(",", ":"))
        except (TypeError, ValueError):
            td_out = "{}"
        return result, conversions, td_out
