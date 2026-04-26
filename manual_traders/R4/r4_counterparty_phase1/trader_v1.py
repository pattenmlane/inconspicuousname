"""Round 4 Phase 2 pilot: react to pre-indexed counterparty prints (tape CSV).

Backtester does not populate state.market_trades; we load
Prosperity4Data/ROUND_4/trades_round_4_day_*.csv once and index by timestamp
(union rows if same ts appears on multiple days — rare overlap).

Rules (from Phase 1 evidence; Phase 2 stress-test under worse fills):
- If any VELVETFRUIT_EXTRACT print at this ts has buyer Mark 67 and price >= L1 ask
  (aggressive buy), buy up to LOT extract at ask (capped).
- If any extract print at this ts has seller Mark 55 and price <= L1 bid, sell
  at bid up to LOT (capped).

HYDROGEL and other VEVs untouched.
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from datamodel import Order, TradingState
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TDIR = REPO / "Prosperity4Data" / "ROUND_4"
EX = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
LIMITS = {
    EX: 200,
    HYDRO: 200,
    **{f"VEV_{k}": 300 for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
}
LOT = 4
COOLDOWN = 15


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


def _touch_extract_signals(ts: int, depth) -> tuple[bool, bool]:
    """Returns (want_buy, want_sell) from tape rows at this timestamp."""
    want_buy, want_sell = False, False
    exd = depth.get(EX)
    if exd is None or not exd.buy_orders or not exd.sell_orders:
        return False, False
    ask = min(exd.sell_orders)
    bid = max(exd.buy_orders)
    for sym, buyer, seller, price, _qty in _TRADES.get(ts, []):
        if sym != EX:
            continue
        if buyer == "Mark 67" and price >= ask - 1e-9:
            want_buy = True
        if seller == "Mark 55" and price <= bid + 1e-9:
            want_sell = True
    return want_buy, want_sell


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
        if EX not in d:
            return o, 0, json.dumps(td)
        ex = d[EX]
        if not ex.buy_orders or not ex.sell_orders:
            return o, 0, json.dumps(td)

        wb, ws = _touch_extract_signals(ts, d)
        pe = state.position.get(EX, 0)
        if tick - last < COOLDOWN:
            return o, 0, json.dumps(td)

        if wb and pe < 80:
            q = min(LOT, LIMITS[EX] - pe, 15)
            if q > 0 and ex.sell_orders:
                o[EX].append(Order(EX, min(ex.sell_orders), q))
                td["last_sig"] = tick
        elif ws and pe > -80:
            q = min(LOT, LIMITS[EX] + pe, 15)
            if q > 0 and ex.buy_orders:
                o[EX].append(Order(EX, max(ex.buy_orders), -q))
                td["last_sig"] = tick

        return o, 0, json.dumps(td)
