"""
Round 4 trader v1 — Phase 2: named-bot + burst conditioning.

- Phase 1: Mark 67 aggressive buy -> small long VELVETFRUIT_EXTRACT (same as v0).
- Phase 2: same-timestamp burst with Mark 01 -> Mark 22 on >=2 symbols -> short VEV_5300 at bid
  (mean fwd5 on 5300 negative near basket bursts; see r4_ph2_mark01_vev5300_burst_window_fwd5.json).

Tape caches outside traderData JSON (module-level).
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
_BURST_TS: dict[int, set[int]] = {}


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


def _burst_timestamps(day: int) -> set[int]:
    """Mark 01 -> Mark 22 ladder: >=3 prints same ts, >=2 distinct symbols."""
    by_ts = _load_trades_by_ts(day)
    out: set[int] = set()
    for ts, rows in by_ts.items():
        if len(rows) < 3:
            continue
        if not any(b == "Mark 01" and s == "Mark 22" for b, s, _, _, _ in rows):
            continue
        if len({r[2] for r in rows}) >= 2:
            out.add(ts)
    return out


EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
VEV_5300 = "VEV_5300"
VEVS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
ALL = [HYDRO, EXTRACT, *VEVS]

POS = {p: (200 if p in (HYDRO, EXTRACT) else 300) for p in ALL}
CLIP_EX = 10
CLIP_5300 = 8


def _bb_ba(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


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
        if day not in _BURST_TS:
            _BURST_TS[day] = _burst_timestamps(day)

        ts = int(state.timestamp)
        tape = _TRADES_CACHE.get(day, {})
        rows = tape.get(ts, [])

        p_ex = state.position.get(EXTRACT, 0)
        p53 = state.position.get(VEV_5300, 0)

        if ts in _BURST_TS.get(day, set()):
            d53 = state.order_depths.get(VEV_5300)
            if d53 is not None and d53.buy_orders and d53.sell_orders:
                bb, _ = _bb_ba(d53)
                if bb is not None:
                    q = min(CLIP_5300, POS[VEV_5300] + p53)
                    if q > 0:
                        out[VEV_5300].append(Order(VEV_5300, int(bb), -q))

        for buyer, _seller, sym, price, _qty in rows:
            if buyer != "Mark 67":
                continue
            od = state.order_depths.get(sym)
            if od is None:
                continue
            bb, ba = _bb_ba(od)
            if bb is None or ba is None or price < ba:
                continue
            if p_ex < POS[EXTRACT]:
                q = min(CLIP_EX, POS[EXTRACT] - p_ex)
                if q > 0:
                    ex_od = state.order_depths.get(EXTRACT)
                    if ex_od and ex_od.sell_orders:
                        ba_ex = min(ex_od.sell_orders)
                        out[EXTRACT].append(Order(EXTRACT, int(math.ceil(float(ba_ex))), q))
            break

        return out, 0, json.dumps(td)
