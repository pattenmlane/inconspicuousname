"""Round 4: v7 with **larger** clips and cap (scale test).

Same logic as v7. LOT=5, MAX_POS=60 vs v7 LOT=3, MAX_POS=36 (still under 300/strike).
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
BURST_MIN = 4
LOT = 5
COOLDOWN = 20
MAX_POS = 60
MIN_HOLD_TICKS = 10


def _build_trades_by_day_ts():
    by: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for csv_day in (1, 2, 3):
        p = TDIR / f"trades_round_4_day_{csv_day}.csv"
        if not p.is_file():
            continue
        with open(p, newline="") as f:
            r = csv.DictReader(f, delimiter=";")
            for row in r:
                ts = int(row["timestamp"])
                by[(csv_day, ts)].append(
                    (
                        str(row["symbol"]),
                        str(row["buyer"]).strip(),
                        str(row["seller"]).strip(),
                        float(row["price"]),
                        int(float(row["quantity"])),
                    )
                )
    return by


_TR = _build_trades_by_day_ts()


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


def _burst_m01_m22(csv_day: int, ts: int) -> bool:
    rows = _TR.get((csv_day, ts), [])
    if len(rows) < BURST_MIN:
        return False
    return any(b == "Mark 01" and s == "Mark 22" for _sym, b, s, _p, _q in rows)


def _join_buy_price(d53) -> int | None:
    if not d53.buy_orders or not d53.sell_orders:
        return None
    bid = max(d53.buy_orders)
    ask = min(d53.sell_orders)
    if ask <= bid:
        return bid
    sp = ask - bid
    if sp >= 2:
        return min(bid + 1, ask)
    return bid


def _improve_sell_price(d53) -> int | None:
    if not d53.buy_orders or not d53.sell_orders:
        return None
    bid = max(d53.buy_orders)
    ask = min(d53.sell_orders)
    if ask <= bid:
        return bid
    sp = ask - bid
    if sp >= 2:
        return max(ask - 1, bid)
    return ask


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
            td["last_active_tick"] = 0
        td["prev_ts"] = ts
        tick = int(td.get("ticks", 0)) + 1
        td["ticks"] = tick
        d_idx = int(td.get("day_idx", 0))
        csv_day = min(3, max(1, d_idx + 1))

        d = state.order_depths
        if S5200 not in d or S5300 not in d:
            return o, 0, json.dumps(td)
        d53 = d[S5300]
        if not d53.buy_orders or not d53.sell_orders:
            return o, 0, json.dumps(td)

        tight = _joint_tight(d)
        burst = _burst_m01_m22(csv_day, ts)
        active = tight and burst
        if active:
            td["last_active_tick"] = tick

        p53 = state.position.get(S5300, 0)
        last = int(td.get("last_sig", 0))
        last_key = td.get("last_burst_key")
        last_active_tick = int(td.get("last_active_tick", 0))

        if not active and p53 > 0:
            if tick - last_active_tick >= MIN_HOLD_TICKS:
                spx = _improve_sell_price(d53)
                if spx is not None:
                    q = min(p53, 40, LIMITS[S5300] + p53)
                    if q > 0:
                        o[S5300].append(Order(S5300, spx, -q))
            return o, 0, json.dumps(td)

        if not active:
            return o, 0, json.dumps(td)

        burst_key = f"{csv_day}:{ts}"
        if burst_key == last_key and (tick - last) < COOLDOWN:
            return o, 0, json.dumps(td)

        if p53 >= MAX_POS:
            return o, 0, json.dumps(td)

        jpx = _join_buy_price(d53)
        if jpx is None:
            return o, 0, json.dumps(td)
        q = min(LOT, LIMITS[S5300] - p53, 12)
        if q > 0:
            o[S5300].append(Order(S5300, jpx, q))
            td["last_sig"] = tick
            td["last_burst_key"] = burst_key

        return o, 0, json.dumps(td)
