"""
vouchers_final_strategy: joint 5200/5300 gate + passive 1-tick BBO improve (v26).

Quote bid at max(bid+1, bid) and min(ask, ask-1) style: actually place at bid+1 and ask-1
when that stays strictly inside the spread, else at best bid/ask. Sizes scale with
tight (both gate legs <=2) vs wide. Local spread only affects size (not price), keeping
the logic small and testable.

VELVET + 10 VEVs only, no HYDROGEL, no spline/LOO.
"""
from __future__ import annotations

import json
from datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_SYMS = [
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]
TRADEABLE = [EXTRACT] + VEV_SYMS
LIMITS = {EXTRACT: 200, **{s: 300 for s in VEV_SYMS}}
GATE = 2


def top_spr(d: OrderDepth | None) -> int | None:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None
    return int(min(d.sell_orders) - max(d.buy_orders))


def joint_tight(state: TradingState) -> bool:
    a = top_spr(state.order_depths.get("VEV_5200"))
    b = top_spr(state.order_depths.get("VEV_5300"))
    if a is None or b is None:
        return False
    return a <= GATE and b <= GATE


class Trader:
    TIGHT_MM = 22
    WIDE_MM = 14
    T_EX = 20
    W_EX = 14

    def run(self, state: TradingState):
        td: dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except json.JSONDecodeError:
                td = {}
        r: dict[str, list[Order]] = {p: [] for p in TRADEABLE}
        t = joint_tight(state)
        mm = self.TIGHT_MM if t else self.WIDE_MM
        exm = self.T_EX if t else self.W_EX

        ex = state.order_depths.get(EXTRACT)
        if ex and ex.buy_orders and ex.sell_orders:
            bb, ba = int(max(ex.buy_orders)), int(min(ex.sell_orders))
            if ba > bb:
                spr = ba - bb
                sz = max(1, int(round(mm * (1.0 - 0.04 * max(0, spr - 2)))))
                r[EXTRACT].extend(
                    self._inside1(EXTRACT, bb, ba, min(sz, exm), state.position.get(EXTRACT, 0), 200)
                )

        for sym in VEV_SYMS:
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            bb, ba = int(max(d.buy_orders)), int(min(d.sell_orders))
            if ba <= bb:
                continue
            spr = ba - bb
            sz = max(1, int(round(mm * (1.0 - 0.05 * max(0, spr - 3)))))
            r[sym].extend(self._inside1(sym, bb, ba, min(sz, mm), state.position.get(sym, 0), 300))
        return r, 0, json.dumps(td)

    def _inside1(
        self, sym: str, bbi: int, bai: int, lot: int, pos: int, lim: int
    ) -> list[Order]:
        o: list[Order] = []
        if bai - bbi < 2:
            return o
        bid_p, ask_p = bbi + 1, bai - 1
        if bid_p < bai and pos < lim:
            o.append(Order(sym, bid_p, min(lot, lim - pos)))
        if ask_p > bbi and pos > -lim:
            o.append(Order(sym, ask_p, -min(lot, lim + pos)))
        return o