"""Round 4: Sonic joint gate + Mark01→Mark22 burst → long VEV_5300 (tape-indexed).

Phase 2/3 tape: M01→M22 multi-row bursts (>=4 trades at ts) showed positive VEV_5300
fwd mids especially under tight surface. Trade CSV has no day column — index trades
by (csv_day, timestamp) using price file day stamps.

- Gate: VEV_5200 and VEV_5300 L1 full spread <= 2.
- Burst: >=4 trade rows at (day, ts) AND any row buyer Mark 01 seller Mark 22.
- Action: buy VEV_5300 at ask up to LOT when gate AND burst (cooldown).
- Flatten long 5300 at bid when gate off or burst off (simple risk).

Uses traderData r4_csv_day: set from first price row day in depth is unavailable — set on
day rollover: when ts < prev, increment day_idx; map day_idx 0->1, 1->2, 2->3 for R4
three-day sequential backtest merge (timestamps reset per day file).
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
LOT = 6
COOLDOWN = 25


def _build_trades_by_day_ts():
    """(csv_day, ts) -> list of (sym, buyer, seller, price, qty) from trades files."""
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

        # Map sequential sim day index to CSV day 1,2,3 (one file per backtest day)
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
        p53 = state.position.get(S5300, 0)
        last = int(td.get("last_sig", 0))

        if (not tight or not burst) and p53 > 0 and d53.buy_orders:
            q = min(p53, 40, LIMITS[S5300] + p53)
            if q > 0:
                o[S5300].append(Order(S5300, max(d53.buy_orders), -q))
            return o, 0, json.dumps(td)

        if not tight or not burst:
            return o, 0, json.dumps(td)

        if tick - last < COOLDOWN:
            return o, 0, json.dumps(td)

        if p53 < 120:
            q = min(LOT, LIMITS[S5300] - p53, 20)
            if q > 0 and d53.sell_orders:
                o[S5300].append(Order(S5300, min(d53.sell_orders), q))
                td["last_sig"] = tick

        return o, 0, json.dumps(td)
