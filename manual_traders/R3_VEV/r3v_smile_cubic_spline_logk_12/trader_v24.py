"""
Round 3 — vouchers_final_strategy thesis only (joint VEV_5200 + VEV_5300 tight gate).

Per round3work/vouchers_final_strategy/STRATEGY.txt:
- Spread = ask1 - bid1; **tight** = (s5200 <= TH) and (s5300 <= TH) with default TH=2.
- In tight regime, short-horizon extract *mid* forward returns are more favorable; use as
  **risk-on** (larger VEV + extract size, lower required edge) vs **wide** (defensive).
- inclineGod: book state (spreads) is the object; Sonic: do not trust edge when either leg is wide.
- This trader does **not** use LOO/cubic-spline IV (legacy per-agent line — dropped).

**PnL focus:** VELVETFRUIT_EXTRACT + 10 VEVs only (no HYDROGEL_PACK).

TTE: round3work/round3description.txt — tape day d => TTE_days = 8 - d, T = TTE_days/365.25
(not used for fair here; we quote from mids).
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

GATE_MAX_SPREAD = 2
EX_SMOOTH = 0.12


def wall_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb, ba = max(depth.buy_orders), min(depth.sell_orders)
    bv, av = depth.buy_orders[bb], -depth.sell_orders[ba]
    tot = bv + av
    if tot <= 0:
        return 0.5 * (bb + ba)
    return (bb * av + ba * bv) / tot


def top_spread(depth: OrderDepth | None) -> int | None:
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None
    return int(min(depth.sell_orders) - max(depth.buy_orders))


def joint_gate_tight(state: TradingState) -> bool:
    """True iff both 5200 and 5300 BBO spreads exist and are <= TH (same as STRATEGY.txt)."""
    s52 = top_spread(state.order_depths.get("VEV_5200"))
    s53 = top_spread(state.order_depths.get("VEV_5300"))
    if s52 is None or s53 is None:
        return False
    return s52 <= GATE_MAX_SPREAD and s53 <= GATE_MAX_SPREAD


class Trader:
    # Risk-on (joint tight)
    TIGHT_TAKE_EDGE = 1.15
    TIGHT_MAKE_EDGE = 0.75
    TIGHT_MM = 24
    TIGHT_TAKE = 28
    TIGHT_EX_EDGE = 2
    TIGHT_EX_LONG_BIAS = 1
    # Risk-off (wide or missing leg)
    WIDE_TAKE_EDGE = 2.35
    WIDE_MAKE_EDGE = 1.1
    WIDE_MM = 12
    WIDE_TAKE = 16
    WIDE_EX_EDGE = 2
    WIDE_EX_LONG_BIAS = 0

    def run(self, state: TradingState):
        td: dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except json.JSONDecodeError:
                td = {}

        result: dict[str, list[Order]] = {p: [] for p in TRADEABLE}

        ex = state.order_depths.get(EXTRACT)
        if ex is None or not ex.buy_orders or not ex.sell_orders:
            return result, 0, json.dumps(td)

        s_mid = wall_mid(ex)
        if s_mid is None or s_mid <= 0:
            return result, 0, json.dumps(td)

        tight = joint_gate_tight(state)
        if tight:
            te, me, mm, tk, ex_edge, ex_bias = (
                self.TIGHT_TAKE_EDGE,
                self.TIGHT_MAKE_EDGE,
                self.TIGHT_MM,
                self.TIGHT_TAKE,
                self.TIGHT_EX_EDGE,
                self.TIGHT_EX_LONG_BIAS,
            )
        else:
            te, me, mm, tk, ex_edge, ex_bias = (
                self.WIDE_TAKE_EDGE,
                self.WIDE_MAKE_EDGE,
                self.WIDE_MM,
                self.WIDE_TAKE,
                self.WIDE_EX_EDGE,
                self.WIDE_EX_LONG_BIAS,
            )

        f_ex = td.get("_fex")
        if f_ex is None:
            f_ex = float(s_mid)
        else:
            f_ex = float(f_ex) + EX_SMOOTH * (float(s_mid) - float(f_ex))
        td["_fex"] = f_ex
        # Optional toy "long extract when risk-on" from STRATEGY layer (2)
        # Smoothed fair in price space; optional +1 shift in risk-on (toy "long extract" from STRATEGY)
        fi = int(round(f_ex)) + ex_bias
        ex_pos = state.position.get(EXTRACT, 0)
        result[EXTRACT].extend(self._quote_extract(ex, ex_pos, fi, ex_edge, max_lot=25))

        for sym in VEV_SYMS:
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            wm = wall_mid(d)
            if wm is None:
                continue
            fair = float(wm)
            pos = state.position.get(sym, 0)
            result[sym].extend(
                self._vev_mm_take(sym, d, pos, fair, te, me, mm, tk, LIMITS[sym])
            )

        return result, 0, json.dumps(td)

    def _vev_mm_take(
        self,
        sym: str,
        depth: OrderDepth,
        pos: int,
        fair: float,
        take_e: float,
        make_e: float,
        mm: int,
        take_sz: int,
        lim: int,
    ) -> list[Order]:
        out: list[Order] = []
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        bb_i, ba_i = int(bb), int(ba)
        if ba_i <= fair - take_e + 1e-9 and pos < lim:
            q = min(take_sz, lim - pos)
            if q > 0:
                out.append(Order(sym, ba_i, q))
        if bb_i >= fair + take_e - 1e-9 and pos > -lim:
            q = min(take_sz, lim + pos)
            if q > 0:
                out.append(Order(sym, bb_i, -q))
        bid_anchor = int(math.floor(fair - make_e))
        ask_anchor = int(math.ceil(fair + make_e))
        bid_p = min(bb_i + 1, bid_anchor)
        bid_p = max(0, bid_p)
        if bid_p < ba_i and pos < lim:
            q = min(mm, lim - pos)
            if q > 0:
                out.append(Order(sym, bid_p, q))
        ask_p = max(ba_i - 1, ask_anchor)
        if ask_p > bb_i and pos > -lim:
            q = min(mm, lim + pos)
            if q > 0:
                out.append(Order(sym, ask_p, -q))
        return out

    def _quote_extract(
        self, depth: OrderDepth, pos: int, fair_i: int, edge: int, max_lot: int = 25
    ) -> list[Order]:
        o: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return o
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        bbi, bai = int(bb), int(ba)
        lim = LIMITS[EXTRACT]
        cap = min(max_lot, lim)
        bid_p = min(bbi + 1, fair_i - edge)
        if bid_p >= 1 and bid_p < bai and pos < lim:
            o.append(Order(EXTRACT, bid_p, min(cap, lim - pos)))
        ask_p = max(bai - 1, fair_i + edge)
        if ask_p > bbi and pos > -lim:
            o.append(Order(EXTRACT, ask_p, -min(cap, lim + pos)))
        return o
