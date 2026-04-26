"""
Round 4 trader_v12: same economics as **trader_v11** with a clearer **tape session day** map.

- **v11** ablates the **M67→Mark 22** ask-suppression *trigger* on tape day **3** (that leg
  hurt worse-fill extract on R4 d3 in sim; see `analysis_outputs/r4_v11_m6722_day3_ablation.txt`).
- **Backtester:** `TestRunner` sets `PROSPERITY4_BACKTEST_DAY` (and we also read optional
  `PROSPERITY4_TAPE_DAY` as an alias) each tick. Per-day `4-1` … `4-3` runs: **v12 == v11**.
- **Merged run** (`4 4` with several days, default timestamp merge, `--no-merge-pnl` off):
  the env stays at **day=1** for all merged ticks, so **v11** would wrongly apply M67→M22
  on the stitched “day 2/3” segments. **v12** infers local session
  `1 + (timestamp // 1_000_000)` when `timestamp >= 1_000_000` (offset **100** per merger),
  and disables the M67→M22 *updates* on **session 3** (third tape in merge order).
- **Single-file day** (ts only `0..999_900` and no env): indistinguishable from day 1 without
  external day id — we default to session **1** (full M67→M22 like v10); set env in the runner
  if you need day 2+ semantics without merge offsets.
"""
from __future__ import annotations

import json
import math
import os
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

M67_LEG_EX_WIN = 250
# After `ResultMerger` stitch, session k (1-based) occupies timestamps
# [ (k-1)*1_000_000 , k*1_000_000 - 100 ]  for k>=2 on top of a day that ends at 999_900.
M67_M22_OFF_SESSION = 3


def _env_tape_session_day() -> int | None:
    """1-based tape day from runner env (preferred)."""
    for k in ("PROSPERITY4_TAPE_DAY", "PROSPERITY4_BACKTEST_DAY"):
        v = os.environ.get(k)
        if v is None or v == "":
            continue
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _tape_session_day(ts: int) -> int:
    """
    1-based session index: env wins; else infer merged multi-tape from timestamp ladder.
    """
    e = _env_tape_session_day()
    if e is not None:
        return e
    t = int(ts)
    if t >= 1_000_000:
        return 1 + (t // 1_000_000)
    return 1


def m67_m22_triggers_may_update(ts: int) -> bool:
    """False on tape/session 3: do not start or extend the M67→M22 ask-suppression leg."""
    return _tape_session_day(int(ts)) != M67_M22_OFF_SESSION


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


def _m67_buyaggr_extract_vs_seller_now(state: TradingState, seller: str) -> bool:
    ex = state.order_depths.get(EXTRACT)
    if ex is None or not ex.buy_orders or not ex.sell_orders:
        return False
    ask1 = float(min(ex.sell_orders))
    ts = int(state.timestamp)
    for t in state.market_trades.get(EXTRACT, []):
        if int(t.timestamp) != ts:
            continue
        if t.buyer != "Mark 67" or t.seller != seller:
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

        if _m67_buyaggr_extract_vs_seller_now(state, "Mark 49"):
            td["_m6749_ex_until"] = ts + M67_LEG_EX_WIN
            td["_m6749_ex_triggers"] = int(td.get("_m6749_ex_triggers", 0)) + 1
        # Do not stack M67→M22 on session 3 (R4 d3 worse-fill regression vs v8 when stacked).
        if m67_m22_triggers_may_update(ts) and _m67_buyaggr_extract_vs_seller_now(state, "Mark 22"):
            td["_m6722_ex_until"] = ts + M67_LEG_EX_WIN
            td["_m6722_ex_triggers"] = int(td.get("_m6722_ex_triggers", 0)) + 1

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
        until49 = int(td.get("_m6749_ex_until", -1))
        until22 = int(td.get("_m6722_ex_until", -1))
        until_m67 = max(until49, until22)
        skip_ex_ask = ts <= until_m67 and not tight
        if skip_ex_ask:
            td["_m67_legs_skip_ask_ticks"] = int(td.get("_m67_legs_skip_ask_ticks", 0)) + 1
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
