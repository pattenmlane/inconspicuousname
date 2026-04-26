"""Round 4 Phase 3 pilot: Sonic joint gate + Mark 67 extract signal (tape CSV index).

Gate (STRATEGY / analyze_vev_5200_5300_tight_gate_r3): L1 spread = min(ask)-max(bid)
on VEV_5200 and VEV_5300; both <= 2.

- When gate **on** and tape shows Mark 67 aggressive **buy** on VELVETFRUIT_EXTRACT,
  buy LOT at ask (same as v1 buy leg).
- When gate **off** and long extract, sell at bid (trim regime risk).
- No Mark 55 leg (Phase 3 focuses gate × informed-style buy).

Trades CSV indexed by timestamp (union across days). market_trades empty in sim.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from datamodel import Order, TradingState
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TDIR = REPO / "Prosperity4Data" / "ROUND_4"
S5200, S5300 = "VEV_5200", "VEV_5300"
EX = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
LIMITS = {
    EX: 200,
    HYDRO: 200,
    **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
}
TH = 2.0
LOT = 4
COOLDOWN = 20


def _l1_spread(d) -> float | None:
    if not d.buy_orders or not d.sell_orders:
        return None
    return float(min(d.sell_orders) - max(d.buy_orders))


def _joint_tight(depth) -> bool:
    if S5200 not in depth or S5300 not in depth:
        return False
    a, b = _l1_spread(depth[S5200]), _l1_spread(depth[S5300])
    if a is None or b is None or a < 0 or b < 0:
        return False
    return a <= TH and b <= TH


def _load_trades_by_ts():
    by_ts: dict[int, list[tuple[str, str, str, float, int]]] = defaultdict(list)
    for day in (1, 2, 3):
        p = TDIR / f"trades_round_4_day_{day}.csv"
        if not p.is_file():
            continue
        with open(p, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                ts = int(row["timestamp"])
                by_ts[ts].append(
                    (
                        str(row["symbol"]),
                        str(row["buyer"]).strip(),
                        str(row["seller"]).strip(),
                        float(row["price"]),
                        int(float(row["quantity"])),
                    )
                )
    return by_ts


_TRADES = _load_trades_by_ts()


def _want_m67_buy(ts: int, depth) -> bool:
    ex = depth.get(EX)
    if ex is None or not ex.buy_orders or not ex.sell_orders:
        return False
    ask = min(ex.sell_orders)
    for sym, buyer, _seller, price, _q in _TRADES.get(ts, []):
        if sym == EX and buyer == "Mark 67" and price >= ask - 1e-9:
            return True
    return False


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}
        o = {p: [] for p in LIMITS}
        ts = state.timestamp
        prev = td.get("prev_ts")
        if prev is not None and ts < prev:
            td["day_idx"] = int(td.get("day_idx", 0)) + 1
        td["prev_ts"] = ts
        tick = int(td.get("ticks", 0)) + 1
        td["ticks"] = tick
        last = int(td.get("last_sig", 0))

        d = state.order_depths
        if EX not in d or S5200 not in d or S5300 not in d:
            return o, 0, json.dumps(td)
        ex = d[EX]
        if not ex.buy_orders or not ex.sell_orders:
            return o, 0, json.dumps(td)

        tight = _joint_tight(d)
        pe = state.position.get(EX, 0)

        if not tight and pe > 0 and ex.buy_orders:
            q = min(pe, 24, LIMITS[EX] + pe)
            if q > 0:
                o[EX].append(Order(EX, max(ex.buy_orders), -q))
            return o, 0, json.dumps(td)

        if not tight:
            return o, 0, json.dumps(td)

        if tick - last < COOLDOWN:
            return o, 0, json.dumps(td)

        if _want_m67_buy(ts, d) and pe < 100:
            q = min(LOT, LIMITS[EX] - pe, 12)
            if q > 0 and ex.sell_orders:
                o[EX].append(Order(EX, min(ex.sell_orders), q))
                td["last_sig"] = tick

        return o, 0, json.dumps(td)
