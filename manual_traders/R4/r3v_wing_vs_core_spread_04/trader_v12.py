"""
Round 4 v12 — combine trader_v1 (Mark 67 buy-aggr extract + act-time Sonic gate) with
Phase 2 Mark 01→Mark 22 **basket burst** follow on VEV_5300 (same lag, same gate).

Burst definition matches r4_phase2_analysis.py / r4_emit_01_22_basket_burst_signals.py:
same (tape_day, timestamp), buyer Mark 01, seller Mark 22, ≥4 trade rows.

Leg A: 1 lot taker buy VELVETFRUIT_EXTRACT on precomputed Mark67 fires (signals_mark67_buy_aggr_extract.json).
Leg B: 1 lot taker buy VEV_5300 on basket-burst fires (signals_01_22_basket_burst_fire.json).

Independent acted sets and position caps (extract 24, voucher strike 300).
"""
from __future__ import annotations

import json
from pathlib import Path

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
H = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
PRODUCTS = [H, U, *[f"VEV_{k}" for k in STRIKES]]

LAG = 100
TIGHT_TH = 2
MAX_LONG_U = 24
MAX_LONG_VEV = 300

_BASE = Path(__file__).resolve().parent

_FIRE_M67: set[int] = set()
_p = _BASE / "outputs" / "phase2" / "signals_mark67_buy_aggr_extract.json"
if _p.is_file():
    for tape_day, local_ts in json.loads(_p.read_text(encoding="utf-8")):
        _FIRE_M67.add((int(tape_day) - 1) * 1_000_000 + int(local_ts) + LAG)

_FIRE_BURST: set[int] = set()
_pb = _BASE / "outputs" / "phase2" / "signals_01_22_basket_burst_fire.json"
if _pb.is_file():
    raw = json.loads(_pb.read_text(encoding="utf-8"))
    for x in raw.get("fires_merged_ts", []):
        _FIRE_BURST.add(int(x))


def _bb_ba(d: OrderDepth | None) -> tuple[int, int] | None:
    if d is None or not d.buy_orders or not d.sell_orders:
        return None
    return int(max(d.buy_orders)), int(min(d.sell_orders))


def _sp(d: OrderDepth | None) -> int | None:
    t = _bb_ba(d)
    if t is None:
        return None
    return t[1] - t[0]


def _joint_tight(state: TradingState) -> bool:
    d52 = state.order_depths.get(VEV_5200)
    d53 = state.order_depths.get(VEV_5300)
    a, b = _sp(d52), _sp(d53)
    if a is None or b is None:
        return False
    return a <= TIGHT_TH and b <= TIGHT_TH


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}
        acted_u: set[int] = set(td.get("acted_u", []))
        acted_v: set[int] = set(td.get("acted_v5300", []))

        orders: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        ts = int(state.timestamp)
        pu = int(state.position.get(U, 0))
        pv = int(state.position.get(VEV_5300, 0))
        tight = _joint_tight(state)

        if ts in _FIRE_M67 and ts not in acted_u and tight and pu < MAX_LONG_U:
            du = state.order_depths.get(U)
            if du and du.sell_orders:
                q = 1
                if pu + q <= MAX_LONG_U:
                    orders[U].append(Order(U, int(min(du.sell_orders)), q))
                    acted_u.add(ts)

        if ts in _FIRE_BURST and ts not in acted_v and tight and pv < MAX_LONG_VEV:
            dv = state.order_depths.get(VEV_5300)
            if dv and dv.sell_orders:
                q = 1
                if pv + q <= MAX_LONG_VEV:
                    orders[VEV_5300].append(Order(VEV_5300, int(min(dv.sell_orders)), q))
                    acted_v.add(ts)

        td["acted_u"] = sorted(acted_u)
        td["acted_v5300"] = sorted(acted_v)
        return orders, 0, json.dumps(td)
