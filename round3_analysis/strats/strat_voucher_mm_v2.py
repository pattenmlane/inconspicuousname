"""Strategy F: Voucher MM with deep liquidity, inventory skew, and aggressive
inside-the-wall posting.

Treat each VEV as its own delta-1 product. The mid mean-reverts on its own
(lag-1 ACF -0.09 to -0.21), so we don't need a theoretical fair — we just
quote tight inside the wall and let the inventory skew flatten us.

Key knobs:
  - SIZE: how many to quote per side per tick.
  - SKEW_TICK: how aggressively to skew (in ticks) per unit of inventory.
  - SOFT_CAP: target max inventory; bias quotes hard once we hit it.
"""
from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TARGETS = {
    "VEV_5100": {"limit": 300, "size": 100, "soft_cap": 200, "skew_per50": 1},
    "VEV_5200": {"limit": 300, "size": 100, "soft_cap": 200, "skew_per50": 1},
    "VEV_5300": {"limit": 300, "size": 100, "soft_cap": 200, "skew_per50": 1},
}


class Trader:
    def run(self, state: TradingState):
        result = {}
        positions = state.position or {}

        for sym, cfg in TARGETS.items():
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
            sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
            if not buys or not sells:
                continue
            bb = max(buys); ba = min(sells); spread = ba - bb
            pos = int(positions.get(sym, 0))
            lim = cfg["limit"]
            size = cfg["size"]
            soft = cfg["soft_cap"]
            skew_per50 = cfg["skew_per50"]

            ords = []
            max_buy = lim - pos
            max_sell = lim + pos

            # ---- Take stale quotes (cross-the-wall) — trade against retail noise ----
            wm = (bb + ba) / 2.0
            for sp in sorted(sells.keys()):
                if max_buy <= 0: break
                if sp <= wm - 1:
                    q = min(sells[sp], max_buy)
                    ords.append(Order(sym, sp, q)); max_buy -= q
                elif sp <= wm and pos < 0:
                    q = min(sells[sp], max_buy, -pos)
                    if q > 0: ords.append(Order(sym, sp, q)); max_buy -= q
            for bp in sorted(buys.keys(), reverse=True):
                if max_sell <= 0: break
                if bp >= wm + 1:
                    q = min(buys[bp], max_sell)
                    ords.append(Order(sym, bp, -q)); max_sell -= q
                elif bp >= wm and pos > 0:
                    q = min(buys[bp], max_sell, pos)
                    if q > 0: ords.append(Order(sym, bp, -q)); max_sell -= q

            # ---- Inside-the-wall passive making ----
            if spread >= 3:
                bid_px = bb + 1; ask_px = ba - 1
            elif spread == 2:
                bid_px = bb; ask_px = ba
            else:
                if ords: result[sym] = ords
                continue

            # Inventory skew: shift quotes by `skew_per50 * (pos // 50)` ticks
            shift = skew_per50 * (pos // 50)
            bid_px = max(bb, bid_px - shift)  # if long, drop bid
            ask_px = max(ask_px - shift, bb + 1)  # if long, drop ask too (be hit-able)
            # Don't cross
            if bid_px >= ask_px:
                bid_px = bb; ask_px = ba

            buy_q = min(size, max_buy, max(0, soft - pos))
            sell_q = min(size, max_sell, max(0, soft + pos))

            if buy_q > 0:
                ords.append(Order(sym, bid_px, buy_q))
            if sell_q > 0:
                ords.append(Order(sym, ask_px, -sell_q))

            if ords:
                result[sym] = ords

        return result, 0, state.traderData or ""
