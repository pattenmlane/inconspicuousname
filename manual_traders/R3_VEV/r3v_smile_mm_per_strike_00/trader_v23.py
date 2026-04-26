"""
Round 3 — **vouchers_final_strategy**: **extract only** when **joint tight gate** is on.

Sonic + STRATEGY: s5200 ≤ 2 and s5300 ≤ 2 (BBO ask−bid) ⇒ **tight** regime. Tape summary
(r3_tight_spread_summary.txt): positive mean K-step **forward mid** on VELVETFRUIT_EXTRACT
when both tight; **v21/v22** showed VEV MM + this gate **hurt** PnL under `worse` vs prior
v9 path — isolate the cash leg to test whether the surface gate is a **-sizing/risk** signal
for extract alone.

- **Tight:** mid-touch market making on **VELVETFRUIT_EXTRACT** only (no VEV orders).
- **Wide:** no orders (same “do not trust small mispricings” as v21 for off-gate).

Mild **long lean** (optional reading of forward-mid): slightly tighter bid / wider ask
when inventory is flat or short so tight regime can accumulate a small long bias; skew still
pushes back when long.

**No HYDROGEL_PACK**, no VEV products.

parent: v22; variant: drop all VEV quoting, keep gate + extract.
"""
from __future__ import annotations

import inspect
import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDERLYING = "VELVETFRUIT_EXTRACT"
GATE_5200 = "VEV_5200"
GATE_5300 = "VEV_5300"
TIGHT_SPREAD_TH = 2

# Mid-touch style on extract: half-spread and clip (grid: 2.2 / 2.4 / 2.6 → 2.4 best on worse)
EXTRACT_HALF = 2.4
SIZE_EXTRACT = 16
# Inventory skew: bias fair toward selling when long (same order of magnitude as v21/v22 VEVs)
SKEW_PER_UNIT = 0.04
# Tight gate is ~10% of bars: small extra bid emphasis when not long (toy directional lean)
LONG_LEAN_TICKS = 0.15

LIMIT_U = 200


def _csv_day_from_backtest_stack() -> int | None:
    for fr in inspect.stack():
        data = fr.frame.f_locals.get("data")
        if data is not None and hasattr(data, "day_num"):
            try:
                return int(getattr(data, "day_num"))
            except (TypeError, ValueError):
                continue
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def book_mid(depth: OrderDepth | None) -> tuple[float, float, float] | None:
    if depth is None:
        return None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if ba <= bb:
        return None
    return float(bb), float(ba), 0.5 * (bb + ba)


def bbo_spread_ticks(depth: OrderDepth | None) -> int | None:
    b = book_mid(depth)
    if b is None:
        return None
    return int(b[1] - b[0])


def joint_tight_gate(
    depths: dict[str, Any], th: int = TIGHT_SPREAD_TH
) -> tuple[bool, int | None, int | None]:
    s5 = bbo_spread_ticks(depths.get(GATE_5200))
    s3 = bbo_spread_ticks(depths.get(GATE_5300))
    if s5 is None or s3 is None:
        return False, s5, s3
    return (s5 <= th and s3 <= th), s5, s3


class Trader:
    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        depths: dict[str, Any] = getattr(state, "order_depths", None) or {}
        positions = getattr(state, "position", None) or {}

        csv_day = _csv_day_from_backtest_stack()
        if csv_day is None:
            csv_day = int(store.get("csv_day_hint", 0))
        store["csv_day_hint"] = csv_day

        if UNDERLYING not in depths:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        bu = book_mid(depths.get(UNDERLYING))
        if bu is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        tight, s5, s3 = joint_tight_gate(depths, TIGHT_SPREAD_TH)
        store["s5200_spread"] = s5
        store["s5300_spread"] = s3
        store["tight_two_leg"] = tight

        if not tight:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        pos_u = int(positions.get(UNDERLYING, 0))
        spr = bu[1] - bu[0]
        # Fair with inventory skew (positive pos -> lower fair, favor selling)
        skew = SKEW_PER_UNIT * (pos_u / max(LIMIT_U, 1))
        fair = bu[2] - skew * spr
        # Long lean: when flat/short, quote bid slightly closer to mid; ask slightly farther
        lean = max(0.0, 1.0 - max(pos_u, 0) / max(LIMIT_U, 1)) * LONG_LEAN_TICKS
        half_b = EXTRACT_HALF - lean
        half_a = EXTRACT_HALF + lean
        bid_x = int(round(fair - half_b))
        ask_x = int(round(fair + half_a))
        bid_x = min(bid_x, int(bu[1]) - 1)
        ask_x = max(ask_x, int(bu[0]) + 1)
        if bid_x >= ask_x:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        ou: list[Order] = []
        qb = min(SIZE_EXTRACT, LIMIT_U - pos_u)
        qs = min(SIZE_EXTRACT, LIMIT_U + pos_u)
        if qb > 0:
            ou.append(Order(UNDERLYING, bid_x, qb))
        if qs > 0:
            ou.append(Order(UNDERLYING, ask_x, -qs))
        if not ou:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        return {UNDERLYING: ou}, 0, json.dumps(store, separators=(",", ":"))
