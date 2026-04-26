"""
Round 4 trader v2 — Phase 3: Sonic joint gate + Mark 67 cue (drops v1 burst short).

Phase 3 tape (r4_ph3_sonic_joint_gate_extract_fwd20.json): inner-join timestamps where
s(VEV_5200)<=2 and s(VEV_5300)<=2 show higher mean K-step forward extract mid than loose regime.

This live rule: only lift VELVETFRUIT_EXTRACT on Mark 67 aggressive buy when BOTH wing L1
spreads are <= 2 at the current tick (same convention as vouchers_final_strategy R3 script).

No burst short 5300 (v1 falsified under worse vs v0).
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
CLIP = 12


def _spread(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return float(min(depth.sell_orders) - max(depth.buy_orders))


def _joint_sonic_gate(state: TradingState) -> bool:
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

        if not _joint_sonic_gate(state):
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
                q = min(CLIP, POS[EXTRACT] - px)
                if q > 0:
                    ex_od = state.order_depths.get(EXTRACT)
                    if ex_od and ex_od.sell_orders:
                        ba_ex = min(ex_od.sell_orders)
                        out[EXTRACT].append(Order(EXTRACT, int(math.ceil(float(ba_ex))), q))
            break

        return out, 0, json.dumps(td)
