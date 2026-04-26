"""
Round 4 v9 — **Mark 67 → Mark 49** extract, **tight at print** (same 12 signals as v8),
lag +100, **no act-time Sonic gate** (ablation vs v8).

Tape check (r4_act_gate_coverage_at_fire.py): joint tight at **fire** time holds on
~50% of these 12 prints anyway — v8’s near-zero SUBMISSION count was dominated by
**worse** matching / ask path, not missing tight rows. v9 measures PnL if we always
taker-buy when the signal fires regardless of 5200/5300 spread at t+LAG.
"""
from __future__ import annotations

import json
from pathlib import Path

from datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
PRODUCTS = [H, U, *[f"VEV_{k}" for k in STRIKES]]

LAG = 100
MAX_LONG = 24

_SIG_PATH = (
    Path(__file__).resolve().parent
    / "outputs"
    / "phase3"
    / "signals_mark67_to_mark49_extract_joint_tight_at_print.json"
)
_FIRE: set[int] = set()
if _SIG_PATH.is_file():
    for tape_day, local_ts in json.loads(_SIG_PATH.read_text(encoding="utf-8")):
        _FIRE.add((int(tape_day) - 1) * 1_000_000 + int(local_ts) + LAG)


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

        if ts in _FIRE and ts not in acted and pu < MAX_LONG:
            du = state.order_depths.get(U)
            if du and du.sell_orders and pu + 1 <= MAX_LONG:
                orders[U].append(Order(U, int(min(du.sell_orders)), 1))
                acted.add(ts)
                td["acted"] = sorted(acted)

        return orders, 0, json.dumps(td)
