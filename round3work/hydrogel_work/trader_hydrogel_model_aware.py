"""
HYDROGEL_PACK — bot-model-aware market maker.

Built on three findings from bot_calibration.txt:

  Bot A quotes: bid = round(FV) - 8,  ask = round(FV) + 8
  Bot B quotes: bid = round(FV-0.5) - 10,  ask = round(FV+0.5) + 10
  Bot C: rare near-FV crossing orders (4% of ticks)

Key insight from calibration:
  wall_mid = (Bot B bid + Bot B ask) / 2 ≈ true_FV  (std 0.35)
  This is already the best FV proxy available from the live book.
  Frankfurt implicitly uses this — and it's correct.

Improvement over plain Frankfurt:
  INVENTORY SKEW — shift the fair value used for passive quoting by
  SKEW_FACTOR * position. When long, fair shifts down: our bid drops
  (we stop buying) and our ask drops (we become cheap to sell against).
  When short, the opposite. Limits blowup during sustained price moves.

  FLATTEN THRESHOLD — above ±FLATTEN_THRESHOLD, suppress passive quoting
  on the side that would worsen the position entirely.

The taking logic is identical to Frankfurt (uses wall_mid as threshold)
to avoid contamination from Bot C displacing Bot A at level 1.

Tunable constants:
  SKEW_FACTOR       — ticks of fair shift per unit of position (default 0.08)
  FLATTEN_THRESHOLD — suppress worsening passive quotes above this (default 150)
"""
from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

SYMBOL = "HYDROGEL_PACK"
POSITION_LIMIT = 200
MAF_BID = 0

SKEW_FACTOR = 0.04       # grid best: +78 vs baseline, safe on volatile days
FLATTEN_THRESHOLD = 150  # never triggered on training data; kept as safety net


class HydrogelModelAwareMM:

    def __init__(self, state: TradingState):
        self.orders: list[Order] = []
        self.pos = int(state.position.get(SYMBOL, 0))
        self.max_buy  = POSITION_LIMIT - self.pos
        self.max_sell = POSITION_LIMIT + self.pos

        depth: OrderDepth | None = state.order_depths.get(SYMBOL)
        # Sort bids high→low, asks low→high — same as Frankfurt
        self.buys:  dict[int, int] = {}
        self.sells: dict[int, int] = {}
        if depth is not None:
            self.buys  = dict(sorted(
                {p: abs(v) for p, v in depth.buy_orders.items()}.items(),
                reverse=True
            ))
            self.sells = dict(sorted(
                {p: abs(v) for p, v in depth.sell_orders.items()}.items()
            ))

        bid_wall = min(self.buys)  if self.buys  else None
        ask_wall = max(self.sells) if self.sells else None

        # wall_mid ≈ true_FV (std 0.35) — best available FV proxy from live book
        self.wall_mid: float | None = (
            (bid_wall + ask_wall) / 2.0
            if bid_wall is not None and ask_wall is not None
            else None
        )

        # Inventory-skewed fair for passive quoting only
        self.fair_skewed: float | None = (
            self.wall_mid - SKEW_FACTOR * self.pos
            if self.wall_mid is not None
            else None
        )

        self.bid_wall = bid_wall
        self.ask_wall = ask_wall

    def _bid(self, price: int, volume: int) -> None:
        q = min(abs(int(volume)), self.max_buy)
        if q <= 0:
            return
        self.orders.append(Order(SYMBOL, int(price), q))
        self.max_buy -= q

    def _ask(self, price: int, volume: int) -> None:
        q = min(abs(int(volume)), self.max_sell)
        if q <= 0:
            return
        self.orders.append(Order(SYMBOL, int(price), -q))
        self.max_sell -= q

    def get_orders(self) -> dict[str, list[Order]]:
        if self.wall_mid is None or self.fair_skewed is None:
            return {SYMBOL: self.orders}

        wm   = self.wall_mid     # FV proxy for taking — identical to Frankfurt
        fair = self.fair_skewed  # inventory-skewed fair for passive quoting

        # ── 1. TAKING — identical logic to Frankfurt ───────────────────────
        for sp, sv in self.sells.items():
            if sp <= wm - 1:
                self._bid(sp, sv)
            elif sp <= wm and self.pos < 0:
                self._bid(sp, min(sv, abs(self.pos)))

        for bp, bv in self.buys.items():
            if bp >= wm + 1:
                self._ask(bp, bv)
            elif bp >= wm and self.pos > 0:
                self._ask(bp, min(bv, self.pos))

        # ── 2. PASSIVE MAKING — skewed fair ───────────────────────────────
        bid_price = int(self.bid_wall) + 1
        ask_price = int(self.ask_wall) - 1

        for bp, bv in self.buys.items():
            if bv > 1 and bp + 1 < fair:
                bid_price = max(bid_price, bp + 1)
                break
            if bp < fair:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in self.sells.items():
            if sv > 1 and sp - 1 > fair:
                ask_price = min(ask_price, sp - 1)
                break
            if sp > fair:
                ask_price = min(ask_price, sp)
                break

        # Suppress worsening side when position is extreme
        if self.pos < FLATTEN_THRESHOLD:
            self._bid(bid_price, self.max_buy)
        if self.pos > -FLATTEN_THRESHOLD:
            self._ask(ask_price, self.max_sell)

        return {SYMBOL: self.orders}


class Trader:
    def bid(self) -> int:
        return int(MAF_BID)

    def run(self, state: TradingState):
        if SYMBOL not in state.order_depths:
            return {}, 0, state.traderData or ""

        mm = HydrogelModelAwareMM(state)
        result = mm.get_orders()

        try:
            print(json.dumps({
                "HYDRO_MODEL": {
                    "ts": state.timestamp,
                    "pos": mm.pos,
                    "wall_mid": mm.wall_mid,
                    "fair_skewed": round(mm.fair_skewed, 2) if mm.fair_skewed else None,
                    "skew": round(SKEW_FACTOR * mm.pos, 2),
                    "n_orders": len(result.get(SYMBOL, [])),
                }
            }, separators=(",", ":")))
        except Exception:
            pass

        return result, 0, state.traderData or ""
