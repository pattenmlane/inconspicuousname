"""
Round 4 trader_v13: **trader_v12** + **Phase 2** extract cue near **Mark01→Mark22** basket bursts.

Tape (`r4_phase2_burst_window_mark20.csv`): **VELVETFRUIT_EXTRACT** prints in ±500 ts of an
offline M01→M22 burst center have **mean mark_20 −0.11** in-window vs **+0.40** out (n=159 vs
1222). **Sim:** omit passive extract **asks** when `burst_near and not joint_gate_tight`
(same gate guard as M67 legs — Sonic surface may dominate when tight). Offline burst centers
match `trader_v2` / `r4_burst_events.csv` (R4 d1–3).
"""
from __future__ import annotations

import bisect
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
BURST_WIN = 500

# Offline centers from analysis_outputs/r4_burst_events.csv (Round 4 days 1–3).
BURST_CENTERS: dict[int, list[int]] = {
    1: [
        4500, 9600, 14800, 16700, 40400, 43400, 50700, 52800, 74100, 85300, 86100, 89800,
        120300, 135900, 145500, 150600, 153000, 170400, 170500, 172500, 190100, 195600,
        235300, 252500, 304000, 310100, 322000, 322600, 350300, 376600, 377900, 399700,
        410400, 428000, 437100, 455700, 460400, 476500, 489700, 519500, 524000, 553100,
        564300, 567200, 568900, 574800, 575800, 588200, 604600, 617900, 618300, 624400,
        636100, 642100, 654200, 674000, 684000, 701200, 703200, 710500, 715000, 720500,
        725000, 735100, 741900, 746300, 754200, 760000, 785800, 798100, 800400, 801800,
        807800, 812600, 848900, 851200, 865400, 901100, 921000, 937300, 943100, 948500,
        950300, 988100,
    ],
    2: [
        13100, 20900, 23500, 29500, 41900, 42900, 45100, 75200, 79300, 102600, 105500,
        129700, 169000, 171800, 189100, 204400, 225500, 239400, 253000, 268900, 275200,
        287100, 287600, 289300, 294600, 316700, 317500, 320800, 321600, 341400, 347800,
        350700, 358500, 362600, 365600, 376300, 389000, 390300, 398200, 410200, 414900,
        415500, 423800, 433100, 443600, 456200, 461200, 483400, 495700, 501900, 508800,
        519600, 554600, 555700, 566700, 575100, 587100, 588500, 593000, 604100, 622600,
        642400, 644500, 666500, 688600, 691100, 711900, 726600, 733200, 748500, 763500,
        764000, 779200, 793900, 808800, 825900, 826400, 836300, 872600, 876500, 898600,
        934900, 948500, 972800, 977300, 977500, 993800,
    ],
    3: [
        32900, 33600, 37000, 40200, 45600, 47200, 54400, 69900, 70100, 70300, 74300, 87300,
        92200, 98900, 118700, 126600, 128700, 132100, 137700, 142700, 144800, 155400,
        166000, 218300, 220800, 222400, 243700, 250200, 253400, 269000, 289400, 290900,
        315800, 317200, 319700, 320200, 328700, 329200, 332600, 346200, 350300, 363100,
        364700, 369800, 370100, 385400, 389100, 391400, 396000, 428900, 431900, 442200,
        451300, 456600, 459800, 483900, 485500, 498100, 511800, 516400, 524000, 529600,
        531500, 535400, 535900, 547600, 551600, 555100, 556400, 565600, 580600, 583100,
        584300, 612700, 614800, 621000, 622500, 624200, 627700, 631300, 631500, 634500,
        642500, 666500, 671800, 671900, 678700, 689900, 699400, 700300, 705400, 708800,
        728500, 730000, 741100, 753700, 766900, 777600, 784100, 785100, 788300, 793100,
        807400, 820200, 826200, 834400, 837500, 839800, 842200, 864100, 868300, 877600,
        878300, 890900, 894300, 918500, 937000, 937400, 943900, 977300, 983300, 997100,
    ],
}

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


def _burst_day_key(ts: int) -> int:
    """Which tape file’s burst centers to use (matches v12 session / env day)."""
    return _tape_session_day(int(ts))


def _near_sorted_ts(ts: int, centers: list[int], win: int) -> bool:
    if not centers:
        return False
    lo = ts - win
    hi = ts + win
    i = bisect.bisect_left(centers, lo)
    return i < len(centers) and centers[i] <= hi


def burst_near_m01_m22(ts: int) -> bool:
    d = _burst_day_key(int(ts))
    return _near_sorted_ts(int(ts), BURST_CENTERS.get(d, []), BURST_WIN)


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
        burst_near = burst_near_m01_m22(ts)
        skip_m67_ask = ts <= until_m67 and not tight
        skip_burst_ask = burst_near and not tight
        if skip_burst_ask:
            td["_burst_ex_skip_ask_ticks"] = int(td.get("_burst_ex_skip_ask_ticks", 0)) + 1
        if skip_m67_ask:
            td["_m67_legs_skip_ask_ticks"] = int(td.get("_m67_legs_skip_ask_ticks", 0)) + 1
        skip_ex_ask = skip_m67_ask or skip_burst_ask
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
