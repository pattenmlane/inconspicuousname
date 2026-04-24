"""
Hold-1 fair probe for Prosperity (Round 3+): buy **1** lot once, then **never trade again**.

After the run, download the website **.log** export and recover server mark / fair as:

    true_fv(t) = profit_and_loss(t) + buy_price

where `buy_price` is the first **SUBMISSION** buy for this product in `tradeHistory`,
and `profit_and_loss` comes from each `activitiesLog` row for that product (same as
`hiddenalphastuff/r2_osmium_fair_278076/export_osmium_fair_log.py`).

**Usage**
1. Set **TARGET_PRODUCT** below to exactly one product string from the round
   (e.g. ``HYDROGEL_PACK``, ``VELVETFRUIT_EXTRACT``, or a voucher like ``VEV_5000``).
2. Upload **only this file** as your submission (or merge `Trader` into your runner).
3. Run **one** historical / sandbox day so the log records the full tape.
4. Export **activitiesLog + tradeHistory**, find `buy_price` from your first buy,
   build ``true_fv`` column for bot calibration / backtests.

If the environment exposes per-tick PnL on `TradingState`, you can also log
``pnl + buy_price`` from `print`; the official export is still the source of truth.

``bid()`` is a stub — set **MAF_BID** if Round 3 includes a market-access auction.
"""

from __future__ import annotations

import json
from datamodel import Listing, Order, OrderDepth, Trade, TradingState

# ---------------------------------------------------------------------------
# Edit this for each upload (one product per submission / or one combined list).
# Must match `listing.product` from the competition `listings` for that instrument.
TARGET_PRODUCT = "VEV_5100"

# Round-3 position limits from round3description.txt (for reference only).
# This script only buys **1** unit.


def _symbol_for_product(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _best_ask(depth: OrderDepth) -> tuple[int, int] | None:
    sells = getattr(depth, "sell_orders", None) or {}
    if not sells:
        return None
    px = min(sells.keys())
    raw = sells[px]
    avail = abs(int(raw))
    return px, avail


def _parse_store(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _submission_buy_price_from_trades(state: TradingState, symbol: str) -> float | None:
    """First fill where we are buyer on ``symbol`` (quantity >= 1)."""
    own = getattr(state, "own_trades", None) or {}
    trades: list[Trade] = own.get(symbol) or []
    for t in trades:
        qty = int(getattr(t, "quantity", 0))
        if qty < 1:
            continue
        buyer = getattr(t, "buyer", None)
        if buyer == "SUBMISSION":
            return float(getattr(t, "price", 0))
    return None


class Trader:

    def run(self, state: TradingState):
        sym = _symbol_for_product(state, TARGET_PRODUCT)
        store = _parse_store(getattr(state, "traderData", None))

        if sym is None:
            # Product not in this run — no-op.
            return {}, 0, json.dumps(store, separators=(",", ":"))

        # `position` is keyed by **symbol** (same key as `order_depths`), not product name.
        pos = int((getattr(state, "position", None) or {}).get(sym, 0))

        # Lock in buy_price from engine trade objects once visible.
        if store.get("buy_price") is None:
            bp = _submission_buy_price_from_trades(state, sym)
            if bp is not None:
                store["buy_price"] = bp
                store["symbol"] = sym

        # Already long >= 1 — hold forever.
        if pos >= 1:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        depth: OrderDepth | None = (getattr(state, "order_depths", None) or {}).get(sym)
        if depth is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ba = _best_ask(depth)
        if ba is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ask_px, avail = ba
        if avail < 1:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        # Single marketable buy: 1 lot at best ask.
        orders = [Order(sym, int(ask_px), 1)]
        return {sym: orders}, 0, json.dumps(store, separators=(",", ":"))