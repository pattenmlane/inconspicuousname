"""
r3v_cross_strike_iv_rank_07 — iteration v33.
vouchers_final_strategy only (no hydrogel, no legacy IV-rank work).

Sonic joint gate: VEV_5200 and VEV_5300 L1 spread both <= 2. When on, two-sided "informed" MM
on VELVETFRUIT_EXTRACT + all 10 VEVs: post bid/ask one tick inside the BBO when spread>=3, else
at touch (inclineGod: book width per name still matters; we do not cross).

STRATEGY.txt caveat: mid forward edge != bid/ask PnL; this is an execution attempt to earn half-spread
when the surface is tight enough to hedge.
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

TH_GATE = 2
WARMUP = 5

# Base size caps per clip (reduced on inventory skew)
Q_EXTRACT = 90
Q_ANCHOR = 100
Q_WING = 50

EX_LIM = 200
VEV_LIM = 300

_EMA = "ema_S"
EMA_N = 12

VOUCHERS = [f"VEV_{k}" for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)]
ANCH = frozenset({"VEV_5200", "VEV_5300"})


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


def _inner(bb: int, ba: int) -> tuple[int, int]:
    if ba - bb >= 3 and bb + 1 < ba - 1:
        return bb + 1, ba - 1
    return bb, ba


def _base(prod: str) -> int:
    if prod in ANCH:
        return Q_ANCHOR
    return Q_WING


def _add_mm(
    out: dict[str, list[Order]],
    sym: str,
    bb: int,
    ba: int,
    pos: int,
    lim: int,
    base: int,
) -> None:
    if ba <= bb:
        return
    bid_p, ask_p = _inner(bb, ba)
    if bid_p >= ask_p:
        bid_p, ask_p = bb, ba
    sk = 1.0
    pabs = abs(pos)
    if pabs > 20:
        sk = max(0.35, 1.0 - (pabs / float(lim)) * 0.75)
    q_b = int(max(0, min(int(base * sk), lim - pos)))
    q_s = int(max(0, min(int(base * sk), lim + pos)))
    if q_b > 0 and bid_p < ba:
        out.setdefault(sym, []).append(Order(sym, bid_p, q_b))
    if q_s > 0 and ask_p > bb:
        out.setdefault(sym, []).append(Order(sym, ask_p, -q_s))


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        ts = int(getattr(state, "timestamp", 0))
        pos: dict[str, int] = getattr(state, "position", None) or {}
        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}

        s_u = _sym(state, "VELVETFRUIT_EXTRACT")
        s5 = _sym(state, "VEV_5200")
        s3 = _sym(state, "VEV_5300")
        if not s_u or not s5 or not s3:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if not _tight(depths, s5, s3):
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
        _add_mm(
            out,
            s_u,
            ubb,
            uba,
            int(pos.get(s_u, 0)),
            EX_LIM,
            Q_EXTRACT,
        )
        for prod in VOUCHERS:
            sy = _sym(state, prod)
            if sy is None:
                continue
            d = depths.get(sy)
            bbi, aai = _ba(d)
            if bbi is None or aai is None:
                continue
            _add_mm(
                out,
                sy,
                bbi,
                aai,
                int(pos.get(sy, 0)),
                VEV_LIM,
                _base(prod),
            )
        return out, 0, json.dumps(td, separators=(",", ":"))
