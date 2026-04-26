"""
Round 4 v5 — **Passive** VEV_5300 bid on Tier-A triggers (same 15 Mark67→Mark22 extract
prints + joint tight at print; lag +100 merged clock).

On the fire step: post one **passive** bid clip (touch or +1 inside if spread≥2).
**Backtest result:** 0 SUBMISSION fills under `--match-trades worse` and `all` — passive
at touch does not match our orders in this engine (same pattern as R3 bid-only v45).

When the gate is **wide** and we hold VEV_5300: **taker flatten** at best bid (unused if
no fills).

No extract leg; no HYDROGEL. Caps per limits.
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

V = VEV_5300
LAG = 100
TIGHT_TH = 2
CLIP_BID = 5
MAX_LONG_5300 = 40
MAX_FLAT = 12

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


def _bid_price(bb: int, ba: int) -> int:
    if ba >= bb + 2:
        return bb + 1
    return bb


class Trader:
    def run(self, state: TradingState):
        try:
            td: dict = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            td = {}
        fired: set[int] = set(td.get("fired_ts", []))

        orders: dict[str, list[Order]] = {p: [] for p in PRODUCTS}
        ts = int(state.timestamp)
        pv = int(state.position.get(V, 0))
        tight = _joint_tight(state)

        if not tight and pv > 0:
            dv = state.order_depths.get(V)
            t = _bb_ba(dv) if dv else None
            if t:
                bb, _ba = t
                q = min(pv, MAX_FLAT)
                if q > 0:
                    orders[V].append(Order(V, bb, -q))

        if ts in _FIRE and ts not in fired and pv < MAX_LONG_5300:
            dv = state.order_depths.get(V)
            t = _bb_ba(dv) if dv else None
            if t:
                bb, ba = t
                q = min(CLIP_BID, MAX_LONG_5300 - pv)
                if q > 0:
                    orders[V].append(Order(V, _bid_price(bb, ba), q))
                    fired.add(ts)
                    td["fired_ts"] = sorted(fired)

        return orders, 0, json.dumps(td)
