"""
Round 4 v2 — Phase 3: same lagged taker follow as v1, but **only** when Sonic joint gate
was **already tight at the Mark 67 print timestamp** (three-way: counterparty × print time × gate).

Signals: outputs/phase3/signals_mark67_buy_aggr_extract_with_tight.json
  list of [tape_day, local_timestamp, joint_tight_at_print].

Still requires joint tight at **execution** time (current book) like v1 — optional redundancy;
primary test is filtering the **164** events down to **~27** tape-consistent tight prints.
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
MAX_LONG = 24
TIGHT_TH = 2

_SIG_PATH = Path(__file__).resolve().parent / "outputs" / "phase3" / "signals_mark67_buy_aggr_extract_with_tight.json"
_FIRE: set[int] = set()
if _SIG_PATH.is_file():
    for row in json.loads(_SIG_PATH.read_text(encoding="utf-8")):
        tape_day, local_ts, tight_print = int(row[0]), int(row[1]), bool(row[2])
        if not tight_print:
            continue
        merged = (tape_day - 1) * 1_000_000 + local_ts + LAG
        _FIRE.add(merged)


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
    a, b = _sp(state.order_depths.get(VEV_5200)), _sp(state.order_depths.get(VEV_5300))
    if a is None or b is None:
        return False
    return a <= TIGHT_TH and b <= TIGHT_TH


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}
        acted: set[int] = set(td.get("acted", []))

        orders: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        ts = int(state.timestamp)
        pu = int(state.position.get(U, 0))

        if ts in _FIRE and ts not in acted and _joint_tight(state) and pu < MAX_LONG:
            du = state.order_depths.get(U)
            if du and du.sell_orders:
                q = 1
                if pu + q <= MAX_LONG:
                    orders[U].append(Order(U, int(min(du.sell_orders)), q))
                    acted.add(ts)
                    td["acted"] = sorted(acted)

        return orders, 0, json.dumps(td)
