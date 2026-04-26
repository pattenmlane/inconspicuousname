"""
Round 3 — joint tight gate + extract momentum + gate-leg VEV (v35).

vouchers_final_strategy/STRATEGY.txt + ORIGINAL_DISCORD_QUOTES.txt only.

- **Gate (Sonic):** BBO spread (ask1-bid1) for VEV_5200 and VEV_5300 both <= 2, same tick.
- **inclineGod:** use spread state (here: joint tight) as the regime, not only mids.
- **Forward extract (K=20) is positive in tight** on our tape
  (tight_regime_forward_extract_k20_by_day.json; matches outputs/r3_tight_spread_summary.txt),
  so in the **tight** regime we **bias extract with short-horizon mid momentum to EMA**
  (toy "long in favorable drift" from STRATEGY item 2), with **small** clips and
  **bid/ask** execution — not mid PnL.
- **VEV_5200 / VEV_5300:** lighter mean reversion to EMA (hedgeable surface) with smaller Q.

**No HYDROGEL_PACK.**
"""
from __future__ import annotations

import json
from typing import Any

from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDER = "VELVETFRUIT_EXTRACT"
GATE = ("VEV_5200", "VEV_5300")
LEGS: tuple[str, str] = ("VEV_5200", "VEV_5300")
TH = 2

# Extract momentum: deviation from EMA; clip when |dev| > EXTRACT_MOM_THR
EMA_U = 40
EXTRACT_MOM_THR = 0.4
Q_EXTRACT = 18
LIM_U = 200

# Gate legs: conservative MR
EMA_V = 100
EDGE_V = 3.5
Q_VEV = 10
LIM_V = 300

WARMUP = 8


def sym_for(state: TradingState, product: str) -> str | None:
    for s, lst in (state.listings or {}).items():
        if getattr(lst, "product", None) == product:
            return s
    return None


def bbo(depth: OrderDepth | None) -> tuple[int | None, int | None, float | None]:
    if depth is None:
        return None, None, None
    bu = depth.buy_orders or {}
    se = depth.sell_orders or {}
    if not bu or not se:
        return None, None, None
    bb = int(max(bu))
    ba = int(min(se))
    if ba < bb:
        return None, None, None
    return bb, ba, 0.5 * (bb + ba)


def spread1(depth: OrderDepth | None) -> int | None:
    bb, ba, _ = bbo(depth)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def ema_bag(bag: dict[str, Any], key: str, w: int, v: float) -> float:
    k = f"e_{key}"
    old = float(bag.get(k, v))
    a = 2.0 / (w + 1.0)
    new = a * v + (1.0 - a) * old
    bag[k] = new
    return new


def merge_orders(orders: dict[str, list[Order]]) -> dict[str, list[Order]]:
    out: dict[str, list[Order]] = {}
    for sym, lst in orders.items():
        acc: dict[int, int] = {}
        for o in lst:
            acc[o.price] = acc.get(o.price, 0) + o.quantity
        merged = [Order(sym, p, q) for p, q in sorted(acc.items()) if q != 0]
        if merged:
            out[sym] = merged
    return out


def joint_tight(depths: dict[str, OrderDepth], sy52: str | None, sy53: str | None) -> bool:
    if sy52 is None or sy53 is None:
        return False
    s1 = spread1(depths.get(sy52))
    s2 = spread1(depths.get(sy53))
    if s1 is None or s2 is None:
        return False
    return s1 <= TH and s2 <= TH


class Trader:
    def run(self, state: TradingState):
        td = getattr(state, "traderData", "") or ""
        try:
            bag: dict[str, Any] = json.loads(td) if td else {}
        except json.JSONDecodeError:
            bag = {}
        if not isinstance(bag, dict):
            bag = {}

        ts = int(getattr(state, "timestamp", 0))
        last = int(bag.get("_last_ts", -1))
        day = int(bag.get("_csv_day", 0))
        if last >= 0 and ts < last:
            day += 1
        bag["_last_ts"] = ts
        bag["_csv_day"] = day

        if ts // 100 < WARMUP:
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", {}) or {}
        pos: dict[str, int] = getattr(state, "position", {}) or {}

        sy_u = sym_for(state, UNDER)
        sy52 = sym_for(state, GATE[0])
        sy53 = sym_for(state, GATE[1])
        if not joint_tight(depths, sy52, sy53):
            return {}, 0, json.dumps(bag, separators=(",", ":"))

        orders: dict[str, list[Order]] = {}

        if sy_u is not None:
            d = depths.get(sy_u)
            bb, ba, mid = bbo(d)
            if mid is not None and bb is not None and ba is not None:
                p0 = int(pos.get(sy_u, 0))
                em = ema_bag(bag, "u", EMA_U, float(mid))
                dev = float(mid) - em
                if dev > EXTRACT_MOM_THR and p0 < LIM_U:
                    q = min(Q_EXTRACT, LIM_U - p0)
                    if q > 0:
                        orders.setdefault(sy_u, []).append(Order(sy_u, int(ba), q))
                elif dev < -EXTRACT_MOM_THR and p0 > -LIM_U:
                    q = min(Q_EXTRACT, LIM_U + p0)
                    if q > 0:
                        orders.setdefault(sy_u, []).append(Order(sy_u, int(bb), -q))

        for prod in LEGS:
            sym = sym_for(state, prod)
            if sym is None:
                continue
            d = depths.get(sym)
            bb, ba, mid = bbo(d)
            if mid is None or bb is None or ba is None:
                continue
            p0 = int(pos.get(sym, 0))
            em = ema_bag(bag, prod, EMA_V, float(mid))
            if mid < em - EDGE_V and p0 < LIM_V:
                q = min(Q_VEV, LIM_V - p0)
                if q > 0:
                    orders.setdefault(sym, []).append(Order(sym, int(ba), q))
            elif mid > em + EDGE_V and p0 > -LIM_V:
                q = min(Q_VEV, LIM_V + p0)
                if q > 0:
                    orders.setdefault(sym, []).append(Order(sym, int(bb), -q))

        return merge_orders(orders), 0, json.dumps(bag, separators=(",", ":"))
