"""
r3v_cross_strike_iv_rank_07 v35 — vouchers_final_strategy.

Same two-sided touch MM as v33 (joint gate) but only VELVETFRUIT_EXTRACT + VEV_5200 + VEV_5300
(wings off) to reduce name-level adverse when posting both sides. Tape shows when gate is on
5200/5300 spreads are exactly 2 (see gate_on_spread_widths_r3.json) — we quote at touch.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH_GATE = 2
WARMUP = 5
Q_EXTRACT = 120
Q_ANCH = 150

EX_LIM = 200
VEV_LIM = 300
_EMA = "ema_S"
EMA_N = 12


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


def _tight(d: dict[str, OrderDepth], a: str, b: str) -> bool:
    ab0, a0 = _ba(d.get(a))
    ab1, a1 = _ba(d.get(b))
    if None in (ab0, a0, ab1, a1):
        return False
    return (a0 - ab0) <= TH_GATE and (a1 - ab1) <= TH_GATE


def _mm(out: dict[str, list[Order]], sym: str, bb: int, ba: int, pos: int, lim: int, base: int) -> None:
    if ba <= bb:
        return
    sk = 1.0
    pa = abs(pos)
    if pa > 15:
        sk = max(0.4, 1.0 - (pa / float(lim)) * 0.7)
    qb = int(max(0, min(int(base * sk), lim - pos)))
    qs = int(max(0, min(int(base * sk), lim + pos)))
    if qb > 0:
        out.setdefault(sym, []).append(Order(sym, bb, qb))
    if qs > 0 and ba > bb:
        out.setdefault(sym, []).append(Order(sym, ba, -qs))


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        s5 = _sym(state, "VEV_5200")
        s3 = _sym(state, "VEV_5300")
        s_u = _sym(state, "VELVETFRUIT_EXTRACT")
        if not s5 or not s3 or not s_u or not _tight(depths, s5, s3):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        du = depths.get(s_u)
        ubb, uba = _ba(du)
        if ubb is None or uba is None:
            return {}, 0, json.dumps(td, separators=(",", ":"))
        s_raw = 0.5 * (ubb + uba)
        pe = td.get(_EMA)
        td[_EMA] = _ema(float(pe) if isinstance(pe, (int, float)) else None, s_raw, EMA_N)
        td["csv_day"] = _day(td, ts, s_raw)
        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        out: dict[str, list[Order]] = {}
        _mm(out, s_u, ubb, uba, int(pos.get(s_u, 0)), EX_LIM, Q_EXTRACT)
        d5 = depths.get(s5)
        b5, a5 = _ba(d5)
        if b5 is not None and a5 is not None:
            _mm(out, s5, b5, a5, int(pos.get(s5, 0)), VEV_LIM, Q_ANCH)
        d3 = depths.get(s3)
        b3, a3 = _ba(d3)
        if b3 is not None and a3 is not None:
            _mm(out, s3, b3, a3, int(pos.get(s3, 0)), VEV_LIM, Q_ANCH)
        return out, 0, json.dumps(td, separators=(",", ":"))
