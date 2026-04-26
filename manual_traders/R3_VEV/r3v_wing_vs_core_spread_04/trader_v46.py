"""
v46 — vouchers_final joint gate (5200+5300 spread <=2) + wide flatten (same as v45).

Tight-regime change vs v45: still **per-contract** BBO (inclineGod) but
- **1-tick improver** on bids when the book has room (join bid+1, else join bid) to
  increase pass-through fills vs resting exactly at the touch with zero worse fills.
- **Recycle longs** at best **ask** with clip capped by *long inventory* (never post
  a sell larger than `max(0, pos)`), so we do not warehouse **net shorts** (v44 pain).

No smile, no HYDROGEL. Extract quoted only if its own spread <= U_SPREAD_MAX.
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

TRADE_VEV = (VEV_5200, VEV_5300, "VEV_5400")

LIMITS = {
    H: 200,
    U: 200,
    **{v: 300 for v in VOUCHERS},
}

TIGHT_S5200_S5300_TH = 2
U_SPREAD_MAX = 6
CLIP = 5
U_CLIP = 4
# Cap **long** inventory per VEV / extract from passive bids in tight (reduces one-way pickoff)
MAX_VEV_LONG = 10
MAX_U_LONG = 16
MAX_FLAT = 12


def _bb_ba(d: OrderDepth) -> tuple[int, int] | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return int(max(d.buy_orders)), int(min(d.sell_orders))


def _sp(d: OrderDepth) -> int | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    bb, ba = _bb_ba(d)  # type: ignore[assignment]
    return int(ba - bb)


def _joint_tight(d52: OrderDepth | None, d53: OrderDepth | None) -> bool:
    if d52 is None or d53 is None:
        return False
    a, b = _sp(d52), _sp(d53)
    if a is None or b is None:
        return False
    return a <= TIGHT_S5200_S5300_TH and b <= TIGHT_S5200_S5300_TH


def _improve_bid_price(bb: int, ba: int) -> int:
    """One tick inside the bid only if spread is at least 2; else at touch (avoid bb+1==ask on 1-tick books)."""
    if ba >= bb + 2:
        return bb + 1
    return bb


def _tight_quotes(
    sym: str,
    d: OrderDepth | None,
    pos: int,
    pos_lim: int,
    long_cap: int,
    bid_clip: int,
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
    # Long: bid up to long_cap, with improver
    bq = min(bid_clip, max(0, long_cap - pos), max(0, pos_lim - pos))
    if bq > 0:
        o.append(Order(sym, _improve_bid_price(bb, ba), bq))
    # Recycle: sell only up to long inventory, at touch ask (no short initiation)
    if pos > 0:
        sq = min(bid_clip, pos)
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

        for v in TRADE_VEV:
            d = state.order_depths.get(v)
            pos = int(state.position.get(v, 0))
            for ord_ in _tight_quotes(
                v, d, pos, LIMITS[v], MAX_VEV_LONG, CLIP
            ):
                orders[v].append(ord_)

        s_u = _sp(du) if du and du.buy_orders and du.sell_orders else None
        if s_u is not None and s_u <= U_SPREAD_MAX and du is not None:
            pu = int(state.position.get(U, 0))
            for ord_ in _tight_quotes(
                U, du, pu, LIMITS[U], MAX_U_LONG, U_CLIP
            ):
                orders[U].append(ord_)

        return orders, 0, json.dumps(td)
