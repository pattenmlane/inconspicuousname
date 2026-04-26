"""
Round 4 v11 — complement of v10: Mark67 buy-aggr extract fires where tape shows
**joint wide** at fire_ts = print_ts+100 (130 of 164; signals_mark67_buy_aggr_extract_wide_at_fire.json).

Tape (mark67_fwd_extract_k20_by_tight_at_fire_summary.json): mean K=20 fwd extract mid
is **lower** when wide-at-fire (1.57) vs tight-at-fire (2.75) — but v10 underperformed v1
because **act-time** gate + `worse` removed many tight-at-fire opportunities. This tests
whether **wide-at-fire** prints (larger v1 subset) recover more of v1's sim PnL.

Same execution as v1: lag+100, act-time Sonic gate, 1-lot taker buy extract.
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

_SIG_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "phase3"
    / "signals_mark67_buy_aggr_extract_wide_at_fire.json"
)
_FIRE: set[int] = set()
if _SIG_PATH.is_file():
    for tape_day, local_ts in json.loads(_SIG_PATH.read_text(encoding="utf-8")):
        _FIRE.add((int(tape_day) - 1) * 1_000_000 + int(local_ts) + LAG)


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
            if du and du.sell_orders and pu + 1 <= MAX_LONG:
                orders[U].append(Order(U, int(min(du.sell_orders)), 1))
                acted.add(ts)
                td["acted"] = sorted(acted)

        return orders, 0, json.dumps(td)
