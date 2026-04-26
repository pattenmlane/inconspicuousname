"""
v26: grid vs v25 — **stricter persistence** (3-tick streak for full size) + stronger **s_prod** penalty.

Same thesis as STRATEGY.txt / vouchers_final_strategy; minor parameter sweep only.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
G5200 = "VEV_5200"
G5300 = "VEV_5300"
VEV_ALL = [f"VEV_{k}" for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]]
SPREAD_TH = 2

BASE_CLIP_VEV = 10
BASE_CLIP_U = 20
WIDE_UNWIND = 3
STREAK_FULL = 3
S_PROD_COEF = 0.12


def _sym(state: TradingState, product: str) -> str | None:
    listings = getattr(state, "listings", {}) or {}
    for sym, lst in listings.items():
        if getattr(lst, "product", None) == product:
            return sym
    return None


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _best_bid_ask(depth: OrderDepth | None):
    if depth is None:
        return None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys)
    ba = min(sells)
    if bb >= ba:
        return None
    return int(bb), int(ba)


def _passive_prices(bb: int, ask: int) -> tuple[int, int]:
    if ask <= bb:
        return bb, ask
    if ask > bb + 1:
        return bb + 1, ask - 1
    return bb, ask


def _vev_limit(product: str) -> int:
    return 200 if product == U else 300


def _s_prod_mult(s_prod: int) -> float:
    s_prod = max(1, min(4, s_prod))
    return max(0.5, 1.0 - S_PROD_COEF * float(s_prod - 1))


def _streak_mult(streak: int) -> float:
    if streak <= 0:
        return 0.0
    if streak >= STREAK_FULL:
        return 1.0
    if streak == 1:
        return 1.0 / 3.0
    return 2.0 / 3.0


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        syms: dict[str, str | None] = {}
        for p in [U, G5200, G5300, *VEV_ALL]:
            syms[p] = _sym(state, p)
        for req in (U, G5200, G5300):
            if syms.get(req) is None:
                return {}, 0, json.dumps(td, separators=(",", ":"))

        depths = getattr(state, "order_depths", None) or {}
        pos = getattr(state, "position", None) or {}

        s5200 = s5300 = -1
        joint_tight = False
        ba52 = _best_bid_ask(depths.get(syms[G5200]))
        ba53 = _best_bid_ask(depths.get(syms[G5300]))
        if ba52 and ba53:
            s5200 = ba52[1] - ba52[0]
            s5300 = ba53[1] - ba53[0]
            joint_tight = s5200 <= SPREAD_TH and s5300 <= SPREAD_TH

        streak = int(td.get("tight_streak", 0) or 0)
        if joint_tight:
            streak = streak + 1
        else:
            streak = 0
        td["tight_streak"] = streak
        s_prod = max(0, s5200) * max(0, s5300) if s5200 >= 0 and s5300 >= 0 else 0
        m_streak = _streak_mult(streak)
        m_sp = _s_prod_mult(s_prod) if joint_tight else 0.0
        size_mult = m_streak * m_sp

        u_ba = _best_bid_ask(depths.get(syms[U]))
        u_mid = 0.0
        if u_ba:
            u_mid = 0.5 * (u_ba[0] + u_ba[1])
        prev = float(td.get("u_prev_mid", 0.0) or 0.0)
        d_u = (u_mid - prev) if prev > 0 else 0.0
        td["u_prev_mid"] = round(u_mid, 4)
        td["d_u"] = round(d_u, 6)
        td["s5200"] = int(s5200)
        td["s5300"] = int(s5300)
        td["s_prod"] = int(s_prod)
        td["tight2"] = int(joint_tight)
        td["th"] = SPREAD_TH
        td["size_mult"] = round(size_mult, 4)
        td["m_streak"] = round(m_streak, 4)
        td["m_sprod"] = round(m_sp, 4)

        orders: dict = {}

        if joint_tight and m_streak > 0.0:
            for p in VEV_ALL + [U]:
                s = syms.get(p)
                if s is None:
                    continue
                ba = _best_bid_ask(depths.get(s))
                if ba is None:
                    continue
                bb, ask = ba
                lim = _vev_limit(p)
                cur = int(pos.get(s, 0))
                clip_u = int(round(BASE_CLIP_U * size_mult))
                clip_v = int(round(BASE_CLIP_VEV * size_mult))
                clip = max(1, clip_u) if p == U else max(1, clip_v)
                clip = min(clip, lim)
                if p == U:
                    if d_u > 0.0:
                        clip = min(lim, clip + int(5 * size_mult) + 1)
                    elif d_u < 0.0:
                        clip = max(1, clip - max(1, int(3 * size_mult)))
                buy_px, sell_px = _passive_prices(bb, ask)
                can_buy = max(0, min(clip, lim - cur))
                can_s = max(0, min(clip, lim + cur))
                if p == U:
                    if d_u > 0.0 and can_buy > 0:
                        orders.setdefault(s, []).append(Order(s, int(buy_px), int(can_buy)))
                    elif d_u < 0.0 and can_s > 0:
                        orders.setdefault(s, []).append(Order(s, int(sell_px), -int(can_s)))
                    elif d_u == 0.0:
                        if can_buy > 0:
                            orders.setdefault(s, []).append(Order(s, int(buy_px), int(can_buy)))
                        if can_s > 0:
                            orders.setdefault(s, []).append(Order(s, int(sell_px), -int(can_s)))
                else:
                    if can_buy > 0:
                        orders.setdefault(s, []).append(Order(s, int(buy_px), int(can_buy)))
                    if can_s > 0:
                        orders.setdefault(s, []).append(Order(s, int(sell_px), -int(can_s)))
        else:
            for p in VEV_ALL + [U]:
                s = syms.get(p)
                if s is None:
                    continue
                ba = _best_bid_ask(depths.get(s))
                if ba is None:
                    continue
                bb, ask = ba
                lim = _vev_limit(p)
                cur = int(pos.get(s, 0))
                qx = min(WIDE_UNWIND, abs(cur))
                if qx <= 0:
                    continue
                if cur > 0:
                    px = ask - 1 if ask > bb + 1 else ask
                    orders.setdefault(s, []).append(Order(s, int(px), -int(qx)))
                else:
                    px = bb + 1 if ask > bb + 1 else bb
                    orders.setdefault(s, []).append(Order(s, int(px), int(qx)))

        return orders, 0, json.dumps(td, separators=(",", ":"))
