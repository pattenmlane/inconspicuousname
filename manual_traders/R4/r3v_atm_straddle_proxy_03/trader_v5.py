"""
Round 4 trader v5 — v4 + Phase-1 second counterparty (Mark 55).

Phase 1 (r4_ph1_adverse_aggressor_fwd20.json): Mark 55 aggressive buys show positive mean fwd20 on
the traded symbol with moderate t-stat. Add a **smaller** extract lift on Mark 55 aggressive buy of
VELVETFRUIT_EXTRACT only, reusing v4 extract spread skip/cap and Sonic-based clip scaling (5 vs 4).

Mark 67 path unchanged from v4. At most one extract order per tick (Mark 67 takes precedence if both
appear in the same tape row batch — rare).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from datamodel import Order, OrderDepth, TradingState

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
_TRADES_CACHE: dict[int, dict[int, list[tuple[str, str, str, int, int]]]] = {}

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
CLIP55_TIGHT = 5
CLIP55_LOOSE = 4
CLIP55_TIGHT_D3 = 4
CLIP55_LOOSE_D3 = 3


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


def _aggressive_buy(od: OrderDepth, price: int) -> bool:
    bb, ba = _bb_ba(od)
    if bb is None or ba is None:
        return False
    return price >= ba


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
        px = state.position.get(EXTRACT, 0)

        if day >= 3:
            lt67, lf67 = CLIP_TIGHT_D3, CLIP_LOOSE_D3
            lt55, lf55 = CLIP55_TIGHT_D3, CLIP55_LOOSE_D3
        else:
            lt67, lf67 = CLIP_TIGHT, CLIP_LOOSE
            lt55, lf55 = CLIP55_TIGHT, CLIP55_LOOSE

        clip67 = lt67 if _sonic_tight(state) else lf67
        clip55 = lt55 if _sonic_tight(state) else lf55

        ex_od = state.order_depths.get(EXTRACT)
        s_ex = _spread(ex_od) if ex_od else None
        if s_ex is not None:
            if s_ex > MAX_EX_SPREAD_SKIP:
                return out, 0, json.dumps(td)
            if s_ex > MAX_EX_SPREAD_CAP:
                clip67 = min(clip67, lf67)
                clip55 = min(clip55, lf55)

        placed = False
        for buyer, _seller, sym, price, _qty in rows:
            if buyer == "Mark 67":
                od = state.order_depths.get(sym)
                if od is None or not _aggressive_buy(od, price):
                    continue
                if px < POS[EXTRACT] and ex_od and ex_od.sell_orders:
                    q = min(clip67, POS[EXTRACT] - px)
                    if q > 0:
                        ba_ex = min(ex_od.sell_orders)
                        out[EXTRACT].append(Order(EXTRACT, int(math.ceil(float(ba_ex))), q))
                        px += q
                        placed = True
                break

        if placed:
            return out, 0, json.dumps(td)

        for buyer, _seller, sym, price, _qty in rows:
            if buyer != "Mark 55" or sym != EXTRACT:
                continue
            if ex_od is None or not _aggressive_buy(ex_od, price):
                continue
            if px < POS[EXTRACT] and ex_od.sell_orders:
                q = min(clip55, POS[EXTRACT] - px)
                if q > 0:
                    ba_ex = min(ex_od.sell_orders)
                    out[EXTRACT].append(Order(EXTRACT, int(math.ceil(float(ba_ex))), q))
            break

        return out, 0, json.dumps(td)
