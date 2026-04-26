"""
Round 3 — vouchers_final_strategy (v28): joint gate + extract momentum + spread-aware inside quotes.

- **Sonic:** trade more size / closer to mid only when VEV_5200 and VEV_5300 BBO spreads
  are both <= 2 (joint tight surface).
- **inclineGod:** per-symbol BBO spread sets improve depth and local size discount.
- **STRATEGY** forward-mid effect: 1-tick **extract** mid momentum (arith mid vs prior tick)
  biases passive extract quotes (toy directional layer; not mid PnL).

No LOO/spline, no HYDROGEL. VELVET + 10 VEVs only.
"""
from __future__ import annotations

import json
import math
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

GATE_TH = 2
F_EX = 0.12
MOMO_THRESH = 0.35
MOMO_MAX = 2


def arith_mid(d: OrderDepth | None) -> float | None:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None
    bb, ba = max(d.buy_orders), min(d.sell_orders)
    return 0.5 * (float(bb) + float(ba))


def top_spr(d: OrderDepth | None) -> int | None:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None
    return int(min(d.sell_orders) - max(d.buy_orders))


def joint_tight(state: TradingState) -> bool:
    a = top_spr(state.order_depths.get("VEV_5200"))
    b = top_spr(state.order_depths.get("VEV_5300"))
    if a is None or b is None:
        return False
    return a <= GATE_TH and b <= GATE_TH


def gate_spread_stress(state: TradingState) -> float:
    """0 when both legs tight; larger when either 5200/5300 is wide (Sonic regime shift)."""
    a = top_spr(state.order_depths.get("VEV_5200"))
    b = top_spr(state.order_depths.get("VEV_5300"))
    if a is None or b is None:
        return 10.0
    return float(max(0, a - GATE_TH) + max(0, b - GATE_TH))


class Trader:
    T_MM = 26
    T_TK = 32
    W_MM = 11
    W_TK = 14
    T_EXL = 24
    W_EXL = 16

    def run(self, state: TradingState):
        td: dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except json.JSONDecodeError:
                td = {}

        r: dict[str, list[Order]] = {p: [] for p in TRADEABLE}
        ex = state.order_depths.get(EXTRACT)
        if ex is None or not ex.buy_orders or not ex.sell_orders:
            return r, 0, json.dumps(td)

        sm = arith_mid(ex)
        if sm is None or sm <= 0.0:
            return r, 0, json.dumps(td)

        prev = td.get("_prev_sm")
        mom = 0.0
        if isinstance(prev, (int, float)):
            mom = float(sm) - float(prev)
        td["_prev_sm"] = float(sm)

        stress = gate_spread_stress(state)
        tight = joint_tight(state)

        f = td.get("_fex")
        f = float(sm) if f is None else float(f) + F_EX * (float(sm) - float(f))
        td["_fex"] = f

        skew = 0
        if float(mom) > MOMO_THRESH:
            skew = min(MOMO_MAX, int(math.floor(0.5 * float(mom) + 0.25)) + 1)
        elif float(mom) < -MOMO_THRESH:
            skew = -min(MOMO_MAX, int(math.floor(0.5 * abs(float(mom)) + 0.25)) + 1)
        fi = int(round(f)) + skew

        ex_pos = state.position.get(EXTRACT, 0)
        t_ex = self.T_EXL if tight else self.W_EXL
        t_ex = max(4, int(round(t_ex * (1.0 - 0.04 * min(6.0, stress)))))
        r[EXTRACT].extend(self._qex(ex, ex_pos, fi, edge=2, max_lot=t_ex))

        mm = self.T_MM if tight else self.W_MM
        tk = self.T_TK if tight else self.W_TK
        mm = max(3, int(round(mm * (1.0 - 0.05 * min(5.0, stress)))))
        tk = max(3, int(round(tk * (1.0 - 0.05 * min(5.0, stress)))))
        if tight and float(mom) > 0.5 * MOMO_THRESH:
            mm = int(round(mm * 1.1))
            tk = int(round(tk * 1.1))

        for sym in VEV_SYMS:
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            bb, ba = int(max(d.buy_orders)), int(min(d.sell_orders))
            if ba <= bb:
                continue
            sp = ba - bb
            loc = max(0, sp - 3)
            mloc = int(round(mm * (1.0 - 0.06 * float(loc))))
            tloc = int(round(tk * (1.0 - 0.06 * float(loc))))
            mloc = max(2, mloc)
            tloc = max(2, tloc)
            pos = state.position.get(sym, 0)
            r[sym].extend(
                self._ins(sym, bb, ba, sp, min(mloc, tloc), pos, LIMITS[sym], two_step=tight)
            )

        return r, 0, json.dumps(td)

    def _qex(
        self, d: OrderDepth, pos: int, fair_i: int, edge: int, max_lot: int
    ) -> list[Order]:
        o: list[Order] = []
        if not d.buy_orders or not d.sell_orders:
            return o
        bbi, bai = int(max(d.buy_orders)), int(min(d.sell_orders))
        L = LIMITS[EXTRACT]
        c = min(max_lot, L)
        bp = min(bbi + 1, fair_i - edge)
        if bp >= 1 and bp < bai and pos < L:
            o.append(Order(EXTRACT, bp, min(c, L - pos)))
        ap = max(bai - 1, fair_i + edge)
        if ap > bbi and pos > -L:
            o.append(Order(EXTRACT, ap, -min(c, L + pos)))
        return o

    def _ins(
        self,
        sym: str,
        bbi: int,
        bai: int,
        spr: int,
        lot: int,
        pos: int,
        lim: int,
        two_step: bool,
    ) -> list[Order]:
        o: list[Order] = []
        if spr < 2:
            return o
        bi = bbi + 1
        ai = bai - 1
        if two_step and spr >= 4:
            bi = min(bbi + 2, bai - 1)
            ai = max(bai - 2, bbi + 1)
        if bi < bai and pos < lim:
            o.append(Order(sym, bi, min(lot, lim - pos)))
        if ai > bbi and pos > -lim:
            o.append(Order(sym, ai, -min(lot, lim + pos)))
        return o