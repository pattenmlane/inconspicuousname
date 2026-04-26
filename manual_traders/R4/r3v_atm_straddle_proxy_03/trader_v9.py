"""
Round 4 trader v9 — v8 + **delayed** Mark 01 buyer-on-VEV_5300 fade (Phase 1 cohort).

Phase 3 showed basket bursts at the print timestamp have **positive** fwd5 on 5300 (conflicts with v1's
same-tick short). Phase 1 population for the K=5 markout is **Mark 01 as buyer on VEV_5300** (n=132 on days 1–3), not
`price>=ask`: on this tape those prints sit at the bid side of the book (passive buyer / mid prints).
v9 **schedules** a small short at `print_ts + DELAY_TICKS` (default 35) for each such row (deduped per ts).

Execution gates (conservative vs v1): Sonic joint tight at **execution** time, VEV_5300 spread <= 6,
clip 4, days 1–2 only (skip day 3 tail). Mark67 extract path unchanged from v8 (incl. M55 lead-lag suppress).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from datamodel import Order, OrderDepth, TradingState

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
_TRADES_CACHE: dict[int, dict[int, list[tuple[str, str, str, int, int]]]] = {}
_SUPPRESS: set[tuple[int, int]] | None = None
# day -> list of timestamps at which to attempt the delayed 5300 fade
_PENDING_5300_FADE: dict[int, list[int]] = defaultdict(list)


def _load_m55_lead_suppress() -> set[tuple[int, int]]:
    global _SUPPRESS
    if _SUPPRESS is not None:
        return _SUPPRESS
    p = Path(__file__).resolve().parent / "analysis_outputs" / "r4_m55_lead_suppress_pairs.json"
    if not p.is_file():
        _SUPPRESS = set()
        return _SUPPRESS
    raw = json.loads(p.read_text())
    _SUPPRESS = {tuple(int(x) for x in pair) for pair in raw}
    return _SUPPRESS

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
VEVS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
ALL = [HYDRO, EXTRACT, *VEVS]

POS = {p: (200 if p in (HYDRO, EXTRACT) else 300) for p in ALL}
TIGHT_TH = 2.0
CLIP_LOOSE = 10
CLIP_TIGHT = 16
CLIP_LOOSE_D3 = 8
CLIP_TIGHT_D3 = 12
MAX_EX_SPREAD_CAP = 10.0
MAX_EX_SPREAD_SKIP = 16.0

DELAY_MARK01_5300 = 35
CLIP_5300_FADE = 4
MAX_S5300_FADE = 6.0


def _spread(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return float(min(depth.sell_orders) - max(depth.buy_orders))


def _sonic_tight(state: TradingState) -> bool:
    d52 = state.order_depths.get(VEV_5200)
    d53 = state.order_depths.get(VEV_5300)
    if d52 is None or d53 is None:
        return False
    s52, s53 = _spread(d52), _spread(d53)
    if s52 is None or s53 is None:
        return False
    return s52 <= TIGHT_TH and s53 <= TIGHT_TH


def _bb_ba(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


def _load_trades_by_ts(day: int) -> dict[int, list[tuple[str, str, str, int, int]]]:
    p = DATA / f"trades_round_4_day_{day}.csv"
    if not p.is_file():
        return {}
    by_ts: dict[int, list[tuple[str, str, str, int, int]]] = {}
    with p.open(newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            ts = int(float(row["timestamp"]))
            by_ts.setdefault(ts, []).append(
                (
                    str(row["buyer"]).strip(),
                    str(row["seller"]).strip(),
                    str(row["symbol"]).strip(),
                    int(float(row["price"])),
                    int(float(row["quantity"])),
                )
            )
    return by_ts


def _schedule_mark01_5300_fades(day: int, rows: list[tuple[str, str, str, int, int]], ts: int) -> None:
    """If Mark 01 bought VEV_5300 at this timestamp (Phase 1 buyer cohort), schedule delayed fade."""
    if day >= 3:
        return
    for buyer, _seller, sym, _price, _qty in rows:
        if buyer != "Mark 01" or sym != VEV_5300:
            continue
        _PENDING_5300_FADE[day].append(ts + DELAY_MARK01_5300)
        break


def _consume_due_fades(day: int, ts: int, state: TradingState, out: dict[str, list[Order]]) -> None:
    """Fire any fades whose scheduled time is <= current ts (backtester skips many integer timestamps)."""
    pend = _PENDING_5300_FADE[day]
    if not pend:
        return
    n_due = sum(1 for t in pend if t <= ts)
    _PENDING_5300_FADE[day] = [t for t in pend if t > ts]
    if n_due == 0:
        return

    if not _sonic_tight(state):
        return
    d53 = state.order_depths.get(VEV_5300)
    if d53 is None or not d53.buy_orders:
        return
    s53 = _spread(d53)
    if s53 is None or s53 > MAX_S5300_FADE:
        return
    bb, _ba = _bb_ba(d53)
    if bb is None:
        return
    p53 = state.position.get(VEV_5300, 0)
    q = min(CLIP_5300_FADE * min(n_due, 3), POS[VEV_5300] + p53)
    if q > 0:
        out[VEV_5300].append(Order(VEV_5300, int(bb), -q))


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        out: dict[str, list[Order]] = {k: [] for k in ALL}
        day = int(getattr(state, "_prosperity4bt_hist_day", 1))
        if day not in _TRADES_CACHE:
            _TRADES_CACHE[day] = _load_trades_by_ts(day)

        ts = int(state.timestamp)
        rows = _TRADES_CACHE.get(day, {}).get(ts, [])

        _consume_due_fades(day, ts, state, out)
        _schedule_mark01_5300_fades(day, rows, ts)

        px = state.position.get(EXTRACT, 0)

        if day >= 3:
            lt, lf = CLIP_TIGHT_D3, CLIP_LOOSE_D3
        else:
            lt, lf = CLIP_TIGHT, CLIP_LOOSE
        clip = lt if _sonic_tight(state) else lf

        ex_od = state.order_depths.get(EXTRACT)
        s_ex = _spread(ex_od) if ex_od else None
        if s_ex is not None:
            if s_ex > MAX_EX_SPREAD_SKIP:
                return out, 0, json.dumps(td)
            if s_ex > MAX_EX_SPREAD_CAP:
                clip = min(clip, lf)

        if (day, ts) in _load_m55_lead_suppress():
            return out, 0, json.dumps(td)

        for buyer, _seller, sym, price, _qty in rows:
            if buyer != "Mark 67":
                continue
            od = state.order_depths.get(sym)
            if od is None:
                continue
            bb, ba = _bb_ba(od)
            if bb is None or ba is None or price < ba:
                continue
            if px < POS[EXTRACT]:
                q = min(clip, POS[EXTRACT] - px)
                if q > 0 and ex_od and ex_od.sell_orders:
                    ba_ex = min(ex_od.sell_orders)
                    out[EXTRACT].append(Order(EXTRACT, int(math.ceil(float(ba_ex))), q))
            break

        return out, 0, json.dumps(td)
