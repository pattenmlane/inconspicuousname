"""
Round 4 v7 — **Dual-leg** on Tier-A triggers (Mark 67→Mark 22 extract + tight at print):
at fire time (lag+100, merged clock), if Sonic gate **still tight**, taker-buy **1 lot
VELVETFRUIT_EXTRACT** and **1 lot VEV_5300** at respective best asks.

Tape follow-up: fwd extract and fwd 5300 mid over K steps are **highly correlated** on
these 15 events (see extract_vev5300_fwd_corr_by_k.csv); joint sum_fwd mean ~+2.2 (K=5)
to ~+3.7 (K=100) on mids — tests whether simultaneous taker legs improve sim PnL vs v3 alone.
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
MAX_LONG_U = 24
MAX_LONG_5300 = 40
TIGHT_TH = 2
V = VEV_5300

_SIG_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "phase3"
    / "signals_mark67_to_mark22_extract_joint_tight_at_print.json"
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
        pv = int(state.position.get(V, 0))

        if ts in _FIRE and ts not in acted and _joint_tight(state):
            du = state.order_depths.get(U)
            dv = state.order_depths.get(V)
            if (
                du
                and du.sell_orders
                and dv
                and dv.sell_orders
                and pu < MAX_LONG_U
                and pv < MAX_LONG_5300
            ):
                orders[U].append(Order(U, int(min(du.sell_orders)), 1))
                orders[V].append(Order(V, int(min(dv.sell_orders)), 1))
                acted.add(ts)
                td["acted"] = sorted(acted)

        return orders, 0, json.dumps(td)
