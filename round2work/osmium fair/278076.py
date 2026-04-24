"""
ASH_COATED_OSMIUM — **one-share probe for live / website runs**

Purpose: as early as possible, **buy exactly 1** contract at the **best ask**
(aggressive), then **stop trading**. Prosperity’s engine marks PnL vs their
internal true fair; you infer that from the **activity / PnL log** on the site
(not from this file).

**Use:** upload this as your submission for the round that lists osmium, run a
short session, download or read their log.

**Not** tuned for local backtesting (no pepper / other products); the
backtester still runs if osmium is in the book.

``traderData``: ``{"filled": true}`` once position is at least 1, so we do not
spam duplicate buys after a partial delay.
"""
from __future__ import annotations

import json

from datamodel import Order, TradingState

SYMBOL = "ASH_COATED_OSMIUM"


class Trader:
    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions = 0

        try:
            raw = state.traderData
            store = json.loads(raw) if (raw and str(raw).strip()) else {}
        except (json.JSONDecodeError, TypeError):
            store = {}

        pos = int(state.position.get(SYMBOL, 0))
        # Only treat JSON boolean true as "done" so arbitrary truthy values cannot strand the probe.
        if pos >= 1 or store.get("filled") is True:
            store["filled"] = True
            return result, conversions, json.dumps(store)

        depth = state.order_depths.get(SYMBOL)
        if not depth or not depth.sell_orders:
            return result, conversions, json.dumps(store)

        best_ask = min(depth.sell_orders.keys())
        ask_sz = abs(int(depth.sell_orders[best_ask]))
        if ask_sz <= 0:
            return result, conversions, json.dumps(store)

        # One aggressive buy; clip to 1 and to available size.
        q = min(1, ask_sz)
        result[SYMBOL] = [Order(SYMBOL, int(best_ask), int(q))]

        return result, conversions, json.dumps(store)