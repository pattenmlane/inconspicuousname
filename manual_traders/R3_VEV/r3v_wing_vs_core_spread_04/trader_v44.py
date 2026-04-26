"""
v44 — round3work/vouchers_final_strategy/ (STRATEGY + ORIGINAL_DISCORD_QUOTES) only.

Sonic: VEV_5200 and VEV_5300 BBO spread both <= 2 at the same time = “tight surface”
where a flow/signal is hedgeable; wide = different regime (stand down).

inclineGod: we read spreads *per* name (here 5200, 5300, then quote using each book’s
BBO) — not mid-only; optional extract spread check before quoting extract.

Implementation (layer-1 “risk filter” from STRATEGY):
- **Tight** joint gate: post **passive** join-the-book two-sided clip on
  VEV_5200, VEV_5300, VEV_5400 (at best bid to buy, best ask to sell) — “trade / quote
  when you can hedge into a tight surface” without smile or IV.
- **Wide**: do not quote those names; one-step **flatten** taker towards zero on
  VEV_5200/5300/5400 and extract if any inventory (avoid straddle in wide book).

No HYDROGEL, no other strikes in this pass (tune TRADE_KS to expand later).
"""
from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
H = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]

# Core surface: gate legs + the adjacent ATM-ish wing for flow (Sonic 5200/5300 + liquid core)
TRADE_VEV = (VEV_5200, VEV_5300, "VEV_5400")

LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in VOUCHERS},
}

TIGHT_S5200_S5300_TH = 2
# Only quote extract if its book is not absurdly wide (inclinGod: per-contract spread)
U_SPREAD_MAX = 8
CLIP = 4
MAX_FLAT = 12


def _bb_ba(d: OrderDepth) -> tuple[int, int] | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return int(max(d.buy_orders)), int(min(d.sell_orders))


def _sp(d: OrderDepth) -> int | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    bb, ba = _bb_ba(d)  # type: ignore[assignment]
    return int(ba - bb)  # type: ignore[operator]


def _joint_tight(d52: OrderDepth | None, d53: OrderDepth | None) -> bool:
    if d52 is None or d53 is None:
        return False
    a, b = _sp(d52), _sp(d53)
    if a is None or b is None:
        return False
    return a <= TIGHT_S5200_S5300_TH and b <= TIGHT_S5200_S5300_TH


def _vev_two_sided(
    sym: str,
    d: OrderDepth | None,
    pos: int,
    lim: int,
    clip: int,
) -> list[Order]:
    if d is None:
        return []
    t = _bb_ba(d)
    if t is None:
        return []
    bb, ba = t
    if ba <= bb:
        return []
    o: list[Order] = []
    bq = min(clip, max(0, lim - pos))
    if bq > 0:
        o.append(Order(sym, bb, bq))
    # Short capacity: position can go to -lim
    sq = min(clip, max(0, pos + lim))
    if sq > 0:
        o.append(Order(sym, ba, -sq))
    return o


def _flatten(sym: str, d: OrderDepth | None, pos: int) -> list[Order]:
    if d is None or not pos or not d.buy_orders or not d.sell_orders:
        return []
    t = _bb_ba(d)
    if t is None:
        return []
    bb, ba = t
    o: list[Order] = []
    if pos > 0:
        q = min(pos, MAX_FLAT)
        o.append(Order(sym, bb, -q))
    else:
        q = min(-pos, MAX_FLAT)
        o.append(Order(sym, ba, q))
    return o


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        orders: dict[str, list[Order]] = {p: [] for p in LIMITS}
        d52 = state.order_depths.get(VEV_5200)
        d53 = state.order_depths.get(VEV_5300)
        du = state.order_depths.get(U)
        tight = _joint_tight(d52, d53)

        if not tight:
            for sym in (*TRADE_VEV, U):
                dep = state.order_depths.get(sym)
                pos = int(state.position.get(sym, 0))
                for ord_ in _flatten(sym, dep, pos):
                    orders[sym].append(ord_)
            return orders, 0, json.dumps(td)

        # Tight: quote core VEVs two-sided; extract if spread on U is not huge
        for v in TRADE_VEV:
            d = state.order_depths.get(v)
            pos = int(state.position.get(v, 0))
            for ord_ in _vev_two_sided(v, d, pos, LIMITS[v], CLIP):
                orders[v].append(ord_)

        s_u = _sp(du) if du and du.buy_orders and du.sell_orders else None
        if s_u is not None and s_u <= U_SPREAD_MAX and du is not None:
            pu = int(state.position.get(U, 0))
            for ord_ in _vev_two_sided(U, du, pu, LIMITS[U], min(CLIP, 6)):
                orders[U].append(ord_)

        return orders, 0, json.dumps(td)
