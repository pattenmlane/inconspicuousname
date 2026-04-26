"""
Round 4 trader v0 — counterparty-aware (Phase 1 evidence).

Offline Phase 1: Mark 67 aggressive buys associate with large positive forward mid on the
traded symbol (see analysis_outputs/r4_ph1_adverse_aggressor_fwd20.json). This bot reads
Round 4 tape trades from CSV (same day as _prosperity4bt_hist_day), keyed by timestamp, and
when it sees Mark 67 aggressively buying (trade price >= best ask on book at that tick),
it lifts a small clip of VELVETFRUIT_EXTRACT (same-direction short-horizon cue).

Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_* per round4description; limits from engine.

No legacy Round 3 smile logic.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from datamodel import Order, OrderDepth, TradingState

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"

# traderData JSON stringifies dict keys — keep tape outside JSON
_TRADES_CACHE: dict[int, dict[int, list[tuple[str, str, str, int, int]]]] = {}

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
VEVS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
ALL = [HYDRO, EXTRACT, *VEVS]

POS = {p: (200 if p in (HYDRO, EXTRACT) else 300) for p in ALL}
CLIP = 12


def _bb_ba(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


def _load_trades_by_ts(day: int) -> dict[int, list[tuple[str, str, str, int, int]]]:
    """timestamp -> list of (buyer, seller, symbol, price, qty)"""
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
        tape = _TRADES_CACHE.get(day, {})
        rows = tape.get(ts, [])

        px = state.position.get(EXTRACT, 0)

        for buyer, _seller, sym, price, _qty in rows:
            if buyer != "Mark 67":
                continue
            od = state.order_depths.get(sym)
            if od is None:
                continue
            bb, ba = _bb_ba(od)
            if bb is None or ba is None:
                continue
            if price < ba:
                continue
            # Aggressive buy on tape: lean long extract (Phase 1 edge)
            if px < POS[EXTRACT]:
                q = min(CLIP, POS[EXTRACT] - px)
                if q > 0:
                    ex_od = state.order_depths.get(EXTRACT)
                    if ex_od and ex_od.sell_orders:
                        ba_ex = min(ex_od.sell_orders)
                        out[EXTRACT].append(Order(EXTRACT, int(math.ceil(float(ba_ex))), q))
            break

        return out, 0, json.dumps(td)
