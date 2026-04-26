"""
HYDROGEL_PACK market maker — Frankfurt wall-mid base + inventory skew.

Same core logic as trader_frankfurt_kelp_style_hydrogel.py with one key addition:
the fair value used for all quoting decisions is shifted proportionally to the
current position (inventory skew). When long, fair shifts down → our bid drops
(we stop buying aggressively) and our ask drops (we become cheaper to sell against).
When short, the opposite. This keeps the strategy from piling into a losing
directional position during sustained price moves.

Also adds a hard flatten threshold: if position exceeds FLATTEN_THRESHOLD,
stop quoting on the side that would worsen it.

Tunable constants:
  SKEW_FACTOR        — ticks of fair-value shift per unit of position.
                       e.g. 0.08 → at pos=+100, fair shifts down 8 ticks.
  FLATTEN_THRESHOLD  — position magnitude above which we stop passive quoting
                       on the worsening side (still take good trades).
"""
from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

SYMBOL = "HYDROGEL_PACK"
POSITION_LIMIT = 200
MAF_BID = 0

SKEW_FACTOR = 0.08          # ticks of fair shift per unit of position
FLATTEN_THRESHOLD = 150     # stop passive quoting on worsening side above this


class HydrogelSkewedMM:

    def __init__(self, state: TradingState):
        self.orders: list[Order] = []
        self.initial_position = int(state.position.get(SYMBOL, 0))
        self.max_allowed_buy = POSITION_LIMIT - self.initial_position
        self.max_allowed_sell = POSITION_LIMIT + self.initial_position

        depth: OrderDepth | None = state.order_depths.get(SYMBOL)
        self.buy_orders: dict[int, int] = {}
        self.sell_orders: dict[int, int] = {}
        if depth is not None:
            self.buy_orders = {
                bp: abs(bv) for bp, bv in sorted(depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            }
            self.sell_orders = {
                sp: abs(sv) for sp, sv in sorted(depth.sell_orders.items(), key=lambda x: x[0])
            }

        self.bid_wall = self.ask_wall = self.wall_mid = None
        try:
            self.bid_wall = min(self.buy_orders.keys())
        except (ValueError, TypeError):
            pass
        try:
            self.ask_wall = max(self.sell_orders.keys())
        except (ValueError, TypeError):
            pass
        if self.bid_wall is not None and self.ask_wall is not None:
            self.wall_mid = (self.bid_wall + self.ask_wall) / 2.0

        self.best_bid = max(self.buy_orders.keys()) if self.buy_orders else None
        self.best_ask = min(self.sell_orders.keys()) if self.sell_orders else None

        # Skewed fair value: shift wall_mid down when long, up when short.
        self.skewed_fair = (
            self.wall_mid - SKEW_FACTOR * self.initial_position
            if self.wall_mid is not None
            else None
        )

    def _bid(self, price: int, volume: int) -> None:
        q = min(abs(int(volume)), self.max_allowed_buy)
        if q <= 0:
            return
        self.orders.append(Order(SYMBOL, int(price), q))
        self.max_allowed_buy -= q

    def _ask(self, price: int, volume: int) -> None:
        q = min(abs(int(volume)), self.max_allowed_sell)
        if q <= 0:
            return
        self.orders.append(Order(SYMBOL, int(price), -q))
        self.max_allowed_sell -= q

    def get_orders(self) -> dict[str, list[Order]]:
        if self.wall_mid is None or self.skewed_fair is None:
            return {SYMBOL: self.orders}

        wm = self.wall_mid
        fair = self.skewed_fair
        pos = self.initial_position

        # ----- 1. Taking — always use raw wall_mid so we don't miss real edge -----
        for sp, sv in self.sell_orders.items():
            if sp <= wm - 1:
                self._bid(sp, sv)
            elif sp <= wm and pos < 0:
                self._bid(sp, min(sv, abs(pos)))

        for bp, bv in self.buy_orders.items():
            if bp >= wm + 1:
                self._ask(bp, bv)
            elif bp >= wm and pos > 0:
                self._ask(bp, min(bv, pos))

        # ----- 2. Making — use skewed fair for quote placement -----
        bid_price = int(self.bid_wall + 1)
        ask_price = int(self.ask_wall - 1)

        for bp, bv in self.buy_orders.items():
            overbidding_price = bp + 1
            if bv > 1 and overbidding_price < fair:
                bid_price = max(bid_price, overbidding_price)
                break
            if bp < fair:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in self.sell_orders.items():
            underbidding_price = sp - 1
            if sv > 1 and underbidding_price > fair:
                ask_price = min(ask_price, underbidding_price)
                break
            if sp > fair:
                ask_price = min(ask_price, sp)
                break

        # Hard flatten: if too long, stop adding buys; if too short, stop adding sells.
        if pos < FLATTEN_THRESHOLD:
            self._bid(bid_price, self.max_allowed_buy)
        if pos > -FLATTEN_THRESHOLD:
            self._ask(ask_price, self.max_allowed_sell)

        return {SYMBOL: self.orders}


class Trader:
    def bid(self) -> int:
        return int(MAF_BID)

    def run(self, state: TradingState):
        if SYMBOL not in state.order_depths:
            return {}, 0, state.traderData or ""

        mm = HydrogelSkewedMM(state)
        result = mm.get_orders()

        try:
            print(
                json.dumps(
                    {
                        "HYDRO_SKEW": {
                            "ts": state.timestamp,
                            "pos": mm.initial_position,
                            "wall_mid": mm.wall_mid,
                            "skewed_fair": mm.skewed_fair,
                            "skew": round(SKEW_FACTOR * mm.initial_position, 2),
                            "n_orders": len(result.get(SYMBOL, [])),
                        }
                    },
                    separators=(",", ":"),
                )
            )
        except Exception:
            pass

        return result, 0, state.traderData or ""
