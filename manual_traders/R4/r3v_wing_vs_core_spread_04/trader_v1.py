"""
Round 4 v1 — tape-conditioned **lagged** follow of Mark 67 buy-aggressive prints on extract.

Phase 1: Mark 67 buy-aggr on VELVETFRUIT_EXTRACT + tight book → positive forward mid (K=20).
Causality-safe execution: act **one tape step (100 time units)** after the print timestamp so the
print is in the past relative to the decision clock (merged log: day d uses timestamps
(d-1)*1_000_000 + local_ts).

Sonic-style gate (from round3work reference): only trade when VEV_5200 and VEV_5300 BBO spreads
are both ≤ 2 at the **current** merged timestamp.

Signals: outputs/phase2/signals_mark67_buy_aggr_extract.json — list of [tape_day, local_timestamp].
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

LAG = 100  # one tape row
MAX_LONG = 24
TIGHT_TH = 2

_SIG_PATH = Path(__file__).resolve().parent / "outputs" / "phase2" / "signals_mark67_buy_aggr_extract.json"
_FIRE: set[int] = set()
if _SIG_PATH.is_file():
    raw = json.loads(_SIG_PATH.read_text(encoding="utf-8"))
    for tape_day, local_ts in raw:
        merged = (int(tape_day) - 1) * 1_000_000 + int(local_ts) + LAG
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
