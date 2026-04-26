"""
Round 4 iteration 11 — Phase 2 bullet 7: inventory skew vs observed sell pressure (Mark 22 on VEVs).

Base: trader_v3 (Sonic joint gate + Mark 67 aggressive buy on VELVETFRUIT_EXTRACT + lift ask).

State:
- EMA(6) of per-timestep count of Mark 22 *aggressive* sells on any VEV_* line
  (trade price at/below that symbol's L1 bid), updated each run when the gate is on.
- "Pressure" = EMA > 0.5 (repeated M22 offer flow into the book).

When long extract (pos > 0) and pressure is high:
- If pos >= POS_SKEW (80): do not add new lifts (avoids piling into M22's supply when already long).
- If pos > POS_SKEW: offer UNWIND_CLIP (18) at best bid to reduce long into that flow.

This is a minimal maker-style skew overlay on the Phase-1/3 taker long.
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
_EMA_M22 = "ema_m22_vev_sellct"
P_M22 = 0.5
POS_SKEW = 80
UNWIND_CLIP = 18
UNWIND_COOLDOWN = 4
EMA_N = 12
M22_EMA_N = 6

EMA_LAST = "ema_m22_last_bucket"
_LAST_FIRE = "last_fire_bucket"
_LAST_UNWIND = "last_unwind_bucket"


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


def _m22_vev_sell_count(state: TradingState, depths: dict[str, OrderDepth]) -> int:
    m = getattr(state, "market_trades", None) or {}
    listings: dict = getattr(state, "listings", None) or {}
    n = 0
    for sym, trades in m.items():
        prod = None
        for s, lst in listings.items():
            if s == sym:
                prod = getattr(lst, "product", None)
                break
        p = str(prod or "")
        if not p.startswith("VEV_"):
            continue
        b, _a = _ba(depths.get(sym))
        if b is None:
            continue
        for tr in trades or []:
            if getattr(tr, "seller", None) == "Mark 22" and int(getattr(tr, "price", 0)) <= int(b):
                n += 1
    return n


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

        bucket = ts // 100
        m22_ct = _m22_vev_sell_count(state, depths)
        le = td.get(EMA_LAST)
        if not isinstance(le, int) or le != bucket:
            pm = td.get(_EMA_M22)
            td[_EMA_M22] = _ema(
                float(pm) if isinstance(pm, (int, float)) else None, float(m22_ct), M22_EMA_N
            )
            td[EMA_LAST] = int(bucket)

        ema_m = float(td[_EMA_M22]) if isinstance(td.get(_EMA_M22), (int, float)) else 0.0
        pressure = ema_m > P_M22

        pos_e = int(pos.get(sym_ex, 0))

        # When already long: trim into M22 VEV offer pressure (maker-style skew)
        if pos_e > 0 and pressure and pos_e > POS_SKEW and ubb is not None:
            leu = td.get(_LAST_UNWIND)
            if not (isinstance(leu, int) and bucket - leu < UNWIND_COOLDOWN):
                qs = min(UNWIND_CLIP, pos_e)
                if qs > 0:
                    td[_LAST_UNWIND] = int(bucket)
                    return {sym_ex: [Order(sym_ex, int(ubb), -qs)]}, 0, json.dumps(td, separators=(",", ":"))

        if not _m67_aggressive_buy_extract(state, sym_ex, int(uba)):
            return {}, 0, json.dumps(td, separators=(",", ":"))

        last = td.get(_LAST_FIRE)
        if isinstance(last, int) and bucket - last < COOLDOWN:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        if pos_e >= POS_SKEW and pressure:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        qb = min(BUY_Q, EX_LIM - pos_e)
        if qb <= 0:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        td[_LAST_FIRE] = int(bucket)
        return {sym_ex: [Order(sym_ex, int(uba), qb)]}, 0, json.dumps(td, separators=(",", ":"))
