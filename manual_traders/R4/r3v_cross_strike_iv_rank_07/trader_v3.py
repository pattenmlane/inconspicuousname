"""
Round 4 iteration 3 — Phase 1/3 synthesis with executable aggression.

Phase 1 edge: Mark 67 aggressive buy on VELVETFRUIT_EXTRACT (trade price >= L1 ask) predicts
short-horizon up-mid on extract.

Phase 3 stack: require Sonic joint gate (VEV_5200 & VEV_5300 L1 spread <= 2) — STRATEGY.txt
recipe; Phase 3 tape shows extract forward k=20 favors tight gate vs loose.

Execution fix vs trader_v1: join-bid at bid+1 is impossible when spread == 1 (bid+1 == ask).
trader_v1 therefore never placed orders on the common 1-tick extract book. Here we lift the
ask (limit buy at best ask) so the order can fill against L1 sells under --match-trades worse.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH_SPREAD = 2
BUY_Q = 24
EX_LIM = 200
COOLDOWN = 6
WARMUP = 5

_EMA_KEY = "ema_S"
EMA_N = 12
_LAST_FIRE = "last_fire_bucket"


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _sym(state: TradingState, prod: str) -> str | None:
    for s, lst in (getattr(state, "listings", {}) or {}).items():
        if getattr(lst, "product", None) == prod:
            return s
    return None


def _ba(d: OrderDepth | None) -> tuple[int | None, int | None]:
    if d is None:
        return None, None
    b, s = getattr(d, "buy_orders", None) or {}, getattr(d, "sell_orders", None) or {}
    if not b or not s:
        return None, None
    return max(b.keys()), min(s.keys())


def _ema(p: float | None, x: float, n: int) -> float:
    if p is None:
        return x
    a = 2.0 / (n + 1.0)
    return a * x + (1.0 - a) * p


def _day(td: dict[str, Any], ts: int, s: float) -> int:
    if ts != 0:
        return int(td.get("csv_day", 0))
    h = td.get("open_S_hist")
    if not isinstance(h, list):
        h = []
    c = round(float(s), 2)
    if not h or abs(float(h[-1]) - c) > 0.25:
        h.append(c)
    td["open_S_hist"] = h[:4]
    return max(0, min(len(h) - 1, 2))


def _sonic_tight(depths: dict[str, OrderDepth], s5: str, s3: str) -> bool:
    b5, a5 = _ba(depths.get(s5))
    b3, a3 = _ba(depths.get(s3))
    if None in (b5, a5, b3, a3):
        return False
    return (a5 - b5) <= TH_SPREAD and (a3 - b3) <= TH_SPREAD


def _m67_aggressive_buy_extract(state: TradingState, sym_ex: str, ask: int) -> bool:
    m = getattr(state, "market_trades", None) or {}
    for tr in m.get(sym_ex, []) or []:
        if getattr(tr, "buyer", None) == "Mark 67" and int(getattr(tr, "price", 0)) >= int(ask):
            return True
    return False


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        sym_ex = _sym(state, "VELVETFRUIT_EXTRACT")
        s520 = _sym(state, "VEV_5200")
        s530 = _sym(state, "VEV_5300")
        if not sym_ex or not s520 or not s530:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        du = depths.get(sym_ex)
        ubb, uba = _ba(du)
        if ubb is None or uba is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        s_raw = 0.5 * (ubb + uba)
        pe = td.get(_EMA_KEY)
        td[_EMA_KEY] = _ema(float(pe) if isinstance(pe, (int, float)) else None, s_raw, EMA_N)
        td["csv_day"] = _day(td, ts, s_raw)

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _sonic_tight(depths, s520, s530):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _m67_aggressive_buy_extract(state, sym_ex, int(uba)):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        bucket = ts // 100
        last = td.get(_LAST_FIRE)
        if isinstance(last, int) and bucket - last < COOLDOWN:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        pos_e = int(pos.get(sym_ex, 0))
        qb = min(BUY_Q, EX_LIM - pos_e)
        if qb <= 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        td[_LAST_FIRE] = int(bucket)
        return {sym_ex: [Order(sym_ex, int(uba), qb)]}, 0, json.dumps(td, separators=(",", ":"))
