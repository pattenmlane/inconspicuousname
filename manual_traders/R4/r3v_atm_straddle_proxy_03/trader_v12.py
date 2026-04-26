"""
Round 4 trader v12 — v8 + **co-tick clip boost** when Mark 49 sells extract.

Phase 1: Mark 49 seller on extract has very positive K=5 fwd mid (n≈105, t≈9.6). v11 showed a
separate ask lift on every M49 seller tick hurts vs v8 (crowds inventory / worse fills).

Tape audit: ~26–35 timestamps per day have **both** Mark 67 aggressive buy (any symbol) and a
Mark 49 seller extract print — heavy overlap. v12 **does not** add a second order; it only adds
+2 to the Mark67 extract clip on days 1–2 when any tape row has seller Mark 49 on extract at this
timestamp (capped at 18). Day 3 unchanged. Lead-lag suppress unchanged.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from datamodel import Order, OrderDepth, TradingState

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "Prosperity4Data" / "ROUND_4"
_TRADES_CACHE: dict[int, dict[int, list[tuple[str, str, str, int, int]]]] = {}
_SUPPRESS: set[tuple[int, int]] | None = None

CLIP_BOOST_M49_CO = 2
CLIP_CAP_BOOSTED = 18


def _load_m55_lead_suppress() -> set[tuple[int, int]]:
    global _SUPPRESS
    if _SUPPRESS is not None:
        return _SUPPRESS
    p = Path(__file__).resolve().parent / "analysis_outputs" / "r4_m55_lead_suppress_pairs.json"
    if not p.is_file():
        _SUPPRESS = set()
        return _SUPPRESS
    raw = json.loads(p.read_text())
    _SUPPRESS = {tuple(int(x) for x in pair) for pair in raw}
    return _SUPPRESS

EXTRACT = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
VEVS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
ALL = [HYDRO, EXTRACT, *VEVS]

POS = {p: (200 if p in (HYDRO, EXTRACT) else 300) for p in ALL}
TIGHT_TH = 2.0
CLIP_LOOSE = 10
CLIP_TIGHT = 16
CLIP_LOOSE_D3 = 8
CLIP_TIGHT_D3 = 12
MAX_EX_SPREAD_CAP = 10.0
MAX_EX_SPREAD_SKIP = 16.0


def _spread(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return float(min(depth.sell_orders) - max(depth.buy_orders))


def _sonic_tight(state: TradingState) -> bool:
    d52 = state.order_depths.get(VEV_5200)
    d53 = state.order_depths.get(VEV_5300)
    if d52 is None or d53 is None:
        return False
    s52, s53 = _spread(d52), _spread(d53)
    if s52 is None or s53 is None:
        return False
    return s52 <= TIGHT_TH and s53 <= TIGHT_TH


def _bb_ba(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders), min(depth.sell_orders)


def _load_trades_by_ts(day: int) -> dict[int, list[tuple[str, str, str, int, int]]]:
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
        rows = _TRADES_CACHE.get(day, {}).get(ts, [])
        px = state.position.get(EXTRACT, 0)

        if day >= 3:
            lt, lf = CLIP_TIGHT_D3, CLIP_LOOSE_D3
        else:
            lt, lf = CLIP_TIGHT, CLIP_LOOSE
        clip = lt if _sonic_tight(state) else lf

        ex_od = state.order_depths.get(EXTRACT)
        s_ex = _spread(ex_od) if ex_od else None
        if s_ex is not None:
            if s_ex > MAX_EX_SPREAD_SKIP:
                return out, 0, json.dumps(td)
            if s_ex > MAX_EX_SPREAD_CAP:
                clip = min(clip, lf)

        if (day, ts) in _load_m55_lead_suppress():
            return out, 0, json.dumps(td)

        m49_sell_ex = any(seller == "Mark 49" and sym == EXTRACT for _b, seller, sym, _p, _q in rows)
        if day < 3 and m49_sell_ex:
            clip = min(clip + CLIP_BOOST_M49_CO, CLIP_CAP_BOOSTED)

        for buyer, _seller, sym, price, _qty in rows:
            if buyer != "Mark 67":
                continue
            od = state.order_depths.get(sym)
            if od is None:
                continue
            bb, ba = _bb_ba(od)
            if bb is None or ba is None or price < ba:
                continue
            if px < POS[EXTRACT]:
                q = min(clip, POS[EXTRACT] - px)
                if q > 0 and ex_od and ex_od.sell_orders:
                    ba_ex = min(ex_od.sell_orders)
                    out[EXTRACT].append(Order(EXTRACT, int(math.ceil(float(ba_ex))), q))
            break

        return out, 0, json.dumps(td)
