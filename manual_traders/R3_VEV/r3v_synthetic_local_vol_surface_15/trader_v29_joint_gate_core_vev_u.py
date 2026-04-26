"""
vouchers_final_strategy — **v29** = v28 with **core strike subset** (no wings).

Rationale: STRATEGY + Sonic focus on **5200/5300** as the hedgeable surface; deep OTM/ITM
wings dominated v28 losses under wider-fill MM. Here we only run the Frankfurt MM on
`VELVETFRUIT_EXTRACT` and **VEV_5000 … VEV_5300** (six names including the two gate legs),
still **no orders** when the joint **5200+5300 ≤2** gate is off.

Same book/FV logic as trader_v28_joint_gate_vev_extract_mm.py.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
GATE_5200 = "VEV_5200"
GATE_5300 = "VEV_5300"
# Core ladder only (drop far wings 4000/4500/6000/6500; keep gate legs + neighbors)
VEV_CORE = [f"VEV_{k}" for k in (5000, 5100, 5200, 5300, 5400, 5500)]

TIGHT_SPREAD_TH = 2
SKEW_FACTOR = 0.02
FLATTEN_THRESHOLD = 200
REGIME_SHIFT = 0.10
POS_U = 200
POS_VEV = 300


@dataclass
class _Book:
    buys: dict[int, int]
    sells: dict[int, int]
    bid_wall: int | None
    ask_wall: int | None
    wall_mid: float | None
    fv_est: float | None


def _read_book(depth: OrderDepth | None) -> _Book | None:
    if depth is None:
        return None
    buys = dict(
        sorted(
            {p: abs(v) for p, v in depth.buy_orders.items()}.items(),
            reverse=True,
        )
    )
    sells = dict(
        sorted({p: abs(v) for p, v in depth.sell_orders.items()}.items())
    )
    if not buys or not sells:
        return None
    bid_wall = min(buys)
    ask_wall = max(sells)
    bid_best = max(buys)
    ask_best = min(sells)
    wall_mid = 0.5 * (float(bid_wall) + float(ask_wall))
    bid2 = bid_wall < bid_best
    ask2 = ask_wall > ask_best
    if bid2 and not ask2:
        fv = wall_mid + REGIME_SHIFT
    elif ask2 and not bid2:
        fv = wall_mid - REGIME_SHIFT
    else:
        fv = wall_mid
    return _Book(
        buys=buys,
        sells=sells,
        bid_wall=bid_wall,
        ask_wall=ask_wall,
        wall_mid=wall_mid,
        fv_est=fv,
    )


def _spread_1tick(depth: OrderDepth | None) -> int | None:
    if depth is None:
        return None
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders.keys())
    ba = min(abs(p) for p in depth.sell_orders.keys())
    if ba <= bb:
        return None
    return int(ba - bb)


def _joint_tight(
    d52: OrderDepth | None, d53: OrderDepth | None, th: int = TIGHT_SPREAD_TH
) -> bool:
    s1 = _spread_1tick(d52)
    s2 = _spread_1tick(d53)
    if s1 is None or s2 is None:
        return False
    return s1 <= th and s2 <= th


def _mm_orders(
    sym: str,
    book: _Book,
    pos: int,
    pos_limit: int,
) -> list[Order]:
    out: list[Order] = []
    max_buy = pos_limit - pos
    max_sell = pos_limit + pos
    if book.fv_est is None or book.wall_mid is None:
        return out
    if book.bid_wall is None or book.ask_wall is None:
        return out

    wm = book.fv_est
    fair = book.fv_est - SKEW_FACTOR * float(pos)

    for sp, sv in book.sells.items():
        if sp <= wm - 1:
            q = min(abs(int(sv)), max_buy)
            if q > 0:
                out.append(Order(sym, int(sp), q))
                max_buy -= q
        elif sp <= wm and pos < 0:
            q = min(abs(int(sv)), max_buy, abs(pos))
            if q > 0:
                out.append(Order(sym, int(sp), q))
                max_buy -= q

    for bp, bv in book.buys.items():
        if bp >= wm + 1:
            q = min(abs(int(bv)), max_sell)
            if q > 0:
                out.append(Order(sym, int(bp), -q))
                max_sell -= q
        elif bp >= wm and pos > 0:
            q = min(abs(int(bv)), max_sell, pos)
            if q > 0:
                out.append(Order(sym, int(bp), -q))
                max_sell -= q

    bid_price = int(book.bid_wall) + 1
    ask_price = int(book.ask_wall) - 1
    for bp, bv in book.buys.items():
        if bv > 1 and bp + 1 < fair:
            bid_price = max(bid_price, bp + 1)
            break
        if bp < fair:
            bid_price = max(bid_price, bp)
            break
    for sp, sv in book.sells.items():
        if sv > 1 and sp - 1 > fair:
            ask_price = min(ask_price, sp - 1)
            break
        if sp > fair:
            ask_price = min(ask_price, sp)
            break
    if pos < FLATTEN_THRESHOLD and max_buy > 0:
        out.append(Order(sym, bid_price, min(max_buy, max(1, min(30, pos_limit // 6)))))
    if pos > -FLATTEN_THRESHOLD and max_sell > 0:
        out.append(Order(sym, ask_price, -min(max_sell, max(1, min(30, pos_limit // 6)))))
    return out


class Trader:
    def run(self, state: TradingState):
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}
        d52 = depths.get(GATE_5200)
        d53 = depths.get(GATE_5300)
        if not _joint_tight(d52, d53):
            return {}, 0, getattr(state, "traderData", "") or ""
        orders_out: dict[str, list[Order]] = {}
        bu = _read_book(depths.get(U))
        if bu is not None:
            for o in _mm_orders(U, bu, int(pos.get(U, 0)), POS_U):
                orders_out.setdefault(U, []).append(o)
        for v in VEV_CORE:
            b = _read_book(depths.get(v))
            if b is None:
                continue
            for o in _mm_orders(v, b, int(pos.get(v, 0)), POS_VEV):
                orders_out.setdefault(v, []).append(o)
        return orders_out, 0, getattr(state, "traderData", "") or ""
