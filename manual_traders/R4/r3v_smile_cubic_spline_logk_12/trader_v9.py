"""
Round 4 trader_v9: **trader_v8** + **inventory-aware** extract ask recycle.

v8 suppresses extract **asks** for **M6749_EX_WIN** ticks after M67→M49 buy-aggr extract
(gate-off), which helped worse/all PnL but can strand **long** extract inventory.

When **skip_ask** would apply but **extract position > M6749_FORCE_ASK_POS**, we
**still place asks** (same `_quote_extract` as v4/v8 unskipped) to unwind inventory.

**R4 days 1–3 grid:** see `analysis_outputs/r4_v9_m6749_force_ask_threshold_sweep.txt`;
thresholds **20–80** all **reduced** worse vs **v8**. Default **`M6749_FORCE_ASK_POS =
LIMITS[EXTRACT] + 1`** never recycles here → **v9 ≡ v8** on this tape; lower the
constant for longer tapes / inventory stress tests.
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
HYDRO = "HYDROGEL_PACK"
TRADEABLE = [HYDRO, EXTRACT] + VEV_SYMS
LIMITS = {
    HYDRO: 200,
    EXTRACT: 200,
    **{s: 300 for s in VEV_SYMS},
}
GATE_MAX_SPREAD = 2
EX_SMOOTH = 0.12
LOCAL_SPR_COEF = 0.12

HYDRO_M38M14_WIN = 400
HYDRO_BASE_EDGE = 3
HYDRO_CAUTION_EDGE = 6
HYDRO_BASE_CLIP = 20
HYDRO_CAUTION_CLIP = 10

M6749_EX_WIN = 250
# Resume extract asks under suppression when position strictly exceeds this (long).
# Default: no recycle on R4 d1–3 (see analysis.json iteration 11); try 60–80 on longer tapes.
M6749_FORCE_ASK_POS = LIMITS[EXTRACT] + 1


def arith_mid(d: OrderDepth) -> float | None:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None
    bb, ba = max(d.buy_orders), min(d.sell_orders)
    return 0.5 * (float(bb) + float(ba))


def top_spread(d: OrderDepth | None) -> int | None:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None
    return int(min(d.sell_orders) - max(d.buy_orders))


def joint_gate_tight(state: TradingState) -> bool:
    s52 = top_spread(state.order_depths.get("VEV_5200"))
    s53 = top_spread(state.order_depths.get("VEV_5300"))
    if s52 is None or s53 is None:
        return False
    return s52 <= GATE_MAX_SPREAD and s53 <= GATE_MAX_SPREAD


def local_spread_penalty(spr: int) -> float:
    return LOCAL_SPR_COEF * max(0.0, float(spr - GATE_MAX_SPREAD))


def _hydro_m38_m14_just_printed(state: TradingState) -> bool:
    ts = int(state.timestamp)
    for t in state.market_trades.get(HYDRO, []):
        if int(t.timestamp) != ts:
            continue
        if t.buyer == "Mark 38" and t.seller == "Mark 14":
            return True
    return False


def _m67_m49_buyaggr_extract_now(state: TradingState) -> bool:
    ex = state.order_depths.get(EXTRACT)
    if ex is None or not ex.buy_orders or not ex.sell_orders:
        return False
    ask1 = float(min(ex.sell_orders))
    ts = int(state.timestamp)
    for t in state.market_trades.get(EXTRACT, []):
        if int(t.timestamp) != ts:
            continue
        if t.buyer != "Mark 67" or t.seller != "Mark 49":
            continue
        if float(t.price) >= ask1 - 1e-9:
            return True
    return False


class Trader:
    TIGHT_BASE_TAKE = 0.72
    TIGHT_BASE_MAKE = 0.48
    TIGHT_MM = 30
    TIGHT_TAKE = 34
    TIGHT_EX_EDGE = 2
    TIGHT_EX_BIAS = 1
    TIGHT_EX_LOT = 30

    WIDE_BASE_TAKE = 1.55
    WIDE_BASE_MAKE = 0.95
    WIDE_MM = 16
    WIDE_TAKE = 20
    WIDE_EX_EDGE = 2
    WIDE_EX_BIAS = 0
    WIDE_EX_LOT = 20

    def run(self, state: TradingState):
        td: dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except json.JSONDecodeError:
                td = {}

        ts = int(state.timestamp)
        if _hydro_m38_m14_just_printed(state):
            td["_hydro_m38_m14_until"] = ts + HYDRO_M38M14_WIN
            td["_hydro_m38_m14_triggers"] = int(td.get("_hydro_m38_m14_triggers", 0)) + 1

        if _m67_m49_buyaggr_extract_now(state):
            td["_m6749_ex_until"] = ts + M6749_EX_WIN
            td["_m6749_ex_triggers"] = int(td.get("_m6749_ex_triggers", 0)) + 1

        result: dict[str, list[Order]] = {p: [] for p in TRADEABLE}

        ex = state.order_depths.get(EXTRACT)
        if ex is None or not ex.buy_orders or not ex.sell_orders:
            return result, 0, json.dumps(td)

        s_mid = arith_mid(ex)
        if s_mid is None or s_mid <= 0:
            return result, 0, json.dumps(td)

        tight = joint_gate_tight(state)
        if tight:
            bt, bm, mm, tk, ex_e, ex_b, ex_l = (
                self.TIGHT_BASE_TAKE,
                self.TIGHT_BASE_MAKE,
                self.TIGHT_MM,
                self.TIGHT_TAKE,
                self.TIGHT_EX_EDGE,
                self.TIGHT_EX_BIAS,
                self.TIGHT_EX_LOT,
            )
        else:
            bt, bm, mm, tk, ex_e, ex_b, ex_l = (
                self.WIDE_BASE_TAKE,
                self.WIDE_BASE_MAKE,
                self.WIDE_MM,
                self.WIDE_TAKE,
                self.WIDE_EX_EDGE,
                self.WIDE_EX_BIAS,
                self.WIDE_EX_LOT,
            )

        f_ex = td.get("_fex")
        if f_ex is None:
            f_ex = float(s_mid)
        else:
            f_ex = float(f_ex) + EX_SMOOTH * (float(s_mid) - float(f_ex))
        td["_fex"] = f_ex
        fi = int(round(f_ex)) + ex_b
        ex_pos = state.position.get(EXTRACT, 0)
        until6749 = int(td.get("_m6749_ex_until", -1))
        skip_ex_ask = ts <= until6749 and not tight
        if skip_ex_ask and ex_pos > M6749_FORCE_ASK_POS:
            skip_ex_ask = False
            td["_m6749_force_ask_inv_ticks"] = int(td.get("_m6749_force_ask_inv_ticks", 0)) + 1
        if skip_ex_ask:
            td["_m6749_skip_ask_ticks"] = int(td.get("_m6749_skip_ask_ticks", 0)) + 1
        result[EXTRACT].extend(
            self._quote_extract(ex, ex_pos, fi, ex_e, max_lot=ex_l, skip_ask=skip_ex_ask)
        )

        for sym in VEV_SYMS:
            d = state.order_depths.get(sym)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            fair = arith_mid(d)
            if fair is None:
                continue
            spr = top_spread(d)
            if spr is None:
                continue
            pen = local_spread_penalty(spr)
            take_e = max(0.35, bt + pen)
            make_e = max(0.2, bm + 0.5 * pen)
            pos = state.position.get(sym, 0)
            result[sym].extend(
                self._vev_mm_take(sym, d, pos, fair, take_e, make_e, mm, tk, LIMITS[sym])
            )

        hd = state.order_depths.get(HYDRO)
        if hd is not None and hd.buy_orders and hd.sell_orders:
            until = int(td.get("_hydro_m38_m14_until", -1))
            hydro_caution = ts <= until
            if hydro_caution:
                td["_hydro_m38_m14_caution_ticks"] = int(td.get("_hydro_m38_m14_caution_ticks", 0)) + 1
            he = HYDRO_CAUTION_EDGE if hydro_caution else HYDRO_BASE_EDGE
            hc = HYDRO_CAUTION_CLIP if hydro_caution else HYDRO_BASE_CLIP
            result[HYDRO].extend(self._quote_hydro(hd, state.position.get(HYDRO, 0), edge=he, max_lot=hc))

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
        self,
        depth: OrderDepth,
        pos: int,
        fair_i: int,
        edge: int,
        max_lot: int = 25,
        skip_ask: bool = False,
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
        if not skip_ask:
            ask_p = max(bai - 1, fair_i + edge)
            if ask_p > bbi and pos > -lim:
                o.append(Order(EXTRACT, ask_p, -min(cap, lim + pos)))
        return o

    def _quote_hydro(self, depth: OrderDepth, pos: int, edge: int = 3, max_lot: int = 20) -> list[Order]:
        o: list[Order] = []
        if not depth.buy_orders or not depth.sell_orders:
            return o
        bb, ba = max(depth.buy_orders), min(depth.sell_orders)
        lim = LIMITS[HYDRO]
        mid = 0.5 * (float(bb) + float(ba))
        fi = int(round(mid))
        cap = min(max_lot, lim)
        bid_p = min(int(bb) + 1, fi - edge)
        if bid_p >= 1 and bid_p < int(ba) and pos < lim:
            o.append(Order(HYDRO, bid_p, min(cap, lim - pos)))
        ask_p = max(int(ba) - 1, fi + edge)
        if ask_p > int(bb) and pos > -lim:
            o.append(Order(HYDRO, ask_p, -min(cap, lim + pos)))
        return o
