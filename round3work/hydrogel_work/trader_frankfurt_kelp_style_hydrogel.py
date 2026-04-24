"""
Frankfurt Hedgehogs (Prosperity 3) — **Rainforest Resin / Kelp-style** market making,
adapted for Prosperity 4 **Round 3** `HYDROGEL_PACK`.

Source of truth:
- Writeup: `Prosperity3Winner/3Writeup.txt` — *Round 1: Market Making* (Resin) and *Kelp*
  (“nearly identical” … wall mid as fair, take favorable trades, then overbid / undercut,
  flatten at fair when inventory skewed).
- Reference code: `Prosperity3Winner/FrankfurtHedgehogs_polished.py` class **`StaticTrader`**
  (Rainforest Resin). **Not** `DynamicTrader` (Kelp + Olivia on the same symbol).

This file only trades **HYDROGEL_PACK**; merge into your multi-product `Trader` if needed.

Position limit from `round3work/round3description.txt`: **200**.
"""

from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

SYMBOL = "HYDROGEL_PACK"
POSITION_LIMIT = 200

# Set if Round 3 introduces a Market Access Fee auction again.
MAF_BID = 0


class HydrogelWallMidMM:
    """Mirrors Frankfurt `StaticTrader` (wall_mid = fair proxy)."""

    def __init__(self, state: TradingState):
        self.state = state
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

    def _bid(self, price: int, volume: int, logging: bool = True) -> None:
        q = min(abs(int(volume)), self.max_allowed_buy)
        if q <= 0:
            return
        if logging:
            pass
        self.orders.append(Order(SYMBOL, int(price), q))
        self.max_allowed_buy -= q

    def _ask(self, price: int, volume: int, logging: bool = True) -> None:
        q = min(abs(int(volume)), self.max_allowed_sell)
        if q <= 0:
            return
        if logging:
            pass
        self.orders.append(Order(SYMBOL, int(price), -q))
        self.max_allowed_sell -= q

    def get_orders(self) -> dict[str, list[Order]]:
        if self.wall_mid is None:
            return {SYMBOL: self.orders}

        wm = self.wall_mid

        # ----- 1. Taking (same structure as Frankfurt StaticTrader) -----
        for sp, sv in self.sell_orders.items():
            if sp <= wm - 1:
                self._bid(sp, sv, logging=False)
            elif sp <= wm and self.initial_position < 0:
                self._bid(sp, min(sv, abs(self.initial_position)), logging=False)

        for bp, bv in self.buy_orders.items():
            if bp >= wm + 1:
                self._ask(bp, bv, logging=False)
            elif bp >= wm and self.initial_position > 0:
                self._ask(bp, min(bv, self.initial_position), logging=False)

        # ----- 2. Making -----
        bid_price = int(self.bid_wall + 1)
        ask_price = int(self.ask_wall - 1)

        for bp, bv in self.buy_orders.items():
            overbidding_price = bp + 1
            if bv > 1 and overbidding_price < wm:
                bid_price = max(bid_price, overbidding_price)
                break
            if bp < wm:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in self.sell_orders.items():
            underbidding_price = sp - 1
            if sv > 1 and underbidding_price > wm:
                ask_price = min(ask_price, underbidding_price)
                break
            if sp > wm:
                ask_price = min(ask_price, sp)
                break

        self._bid(bid_price, self.max_allowed_buy)
        self._ask(ask_price, self.max_allowed_sell)

        return {SYMBOL: self.orders}


class Trader:
    def bid(self) -> int:
        return int(MAF_BID)

    def run(self, state: TradingState):
        if SYMBOL not in state.order_depths:
            return {}, 0, state.traderData or ""

        mm = HydrogelWallMidMM(state)
        result = mm.get_orders()

        # Optional: compact debug line (comment out for production upload).
        try:
            print(
                json.dumps(
                    {
                        "HYDRO_MM": {
                            "ts": state.timestamp,
                            "pos": int(state.position.get(SYMBOL, 0)),
                            "wall_mid": mm.wall_mid,
                            "n_orders": len(result.get(SYMBOL, [])),
                        }
                    },
                    separators=(",", ":"),
                )
            )
        except Exception:
            pass

        return result, 0, state.traderData or ""
