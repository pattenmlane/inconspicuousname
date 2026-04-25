"""
VELVETFRUIT_EXTRACT — bot-model-aware market maker.

Built on three findings from bot_calibration.txt:

  Bot A quotes: bid = floor(FV - 0.1) - 2,  ask = ceil(FV + 0.1) + 2
  Bot B quotes: floor(FV) - 3 / ceil(FV) + 3, conditional on frac(FV)
  Bot C: rare near-FV crossing orders (~4% of ticks)

Key insight from calibration:
  wall_mid = (min_bids + max_asks) / 2 ≈ true_FV  (std 0.23 — very tight)
  This is the best FV proxy available from the live book.

VEV characteristics vs hydrogel:
  - Bot A spread is ±2–3 ticks (vs hydrogel ±8 ticks) — MUCH tighter
  - FV volatility is std=7.5 (vs hydrogel std=30) — calmer
  - Taking opportunities (ask ≤ wm-1, bid ≥ wm+1) are rarer because the spread is tight
  - Passive making inside Bot A's spread is the primary profit engine

Strategy:
  TAKING — same Frankfurt logic: take any ask ≤ wm-1 or bid ≥ wm+1
  MAKING — overbid 1 tick above Bot B's outer quote (when present), or just inside Bot A

  INVENTORY SKEW — shift the fair value used for passive quoting by
  SKEW_FACTOR * position. When long, fair shifts down: our bid drops
  and our ask drops. Limits blowup during sustained price moves.

  FLATTEN THRESHOLD — suppress passive quoting on worsening side above this.

Tunable constants:
  SKEW_FACTOR       — ticks of fair shift per unit of position (default 0.02)
  FLATTEN_THRESHOLD — suppress worsening passive quotes above this (default 150)
"""
from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

SYMBOL = "VELVETFRUIT_EXTRACT"
POSITION_LIMIT = 200
MAF_BID = 0

SKEW_FACTOR = 0.00       # grid best: no skew — VEV is calm enough that inventory self-corrects
FLATTEN_THRESHOLD = 150  # suppress worsening passive quotes above this


class VevModelAwareMM:

    def __init__(self, state: TradingState):
        self.orders: list[Order] = []
        self.pos = int(state.position.get(SYMBOL, 0))
        self.max_buy  = POSITION_LIMIT - self.pos
        self.max_sell = POSITION_LIMIT + self.pos

        depth: OrderDepth | None = state.order_depths.get(SYMBOL)
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

        # wall_mid ≈ true_FV (std 0.23) — best FV proxy from live book
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

        wm   = self.wall_mid     # FV proxy for taking — unbiased
        fair = self.fair_skewed  # inventory-skewed fair for passive quoting

        # ── 1. TAKING — same Frankfurt logic ───────────────────────────────
        # With ±2–3 tick spread, these fire only when a stale quote is in the book
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

        # ── 2. PASSIVE MAKING — skewed fair ────────────────────────────────
        # Start from just inside the wall (Bot B outer quote or Bot A if B absent)
        bid_price = int(self.bid_wall) + 1
        ask_price = int(self.ask_wall) - 1

        # Walk the buy side: place our bid as high as possible below fair
        for bp, bv in self.buys.items():
            if bv > 1 and bp + 1 < fair:
                bid_price = max(bid_price, bp + 1)
                break
            if bp < fair:
                bid_price = max(bid_price, bp)
                break

        # Walk the sell side: place our ask as low as possible above fair
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

        mm = VevModelAwareMM(state)
        result = mm.get_orders()

        try:
            print(json.dumps({
                "VEV_MODEL": {
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
