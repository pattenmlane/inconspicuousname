"""
vouchers_final_strategy only (no legacy RV-IV / butterfly / family-13 logic).

(Execution) Must support one-tick-wide books: use touch bid/ask if spread==1, else bid+1/ask-1.
Earlier revision incorrectly skipped 1-tick books when the joint gate was on.

- Sonic joint gate: VEV_5200 and VEV_5300 BBO spread both <= TH (2 ticks) at this timestamp
  = \"tight surface\" / risk-on; else \"wide\" = different regime (reduce noise; execution dominates).
- inclineGod: use spread levels and spread^52 * spread^53 as a scalar *state* in traderData
  (book width co-tightness), not only mid correlation.
- Trade VELVETFRUIT_EXTRACT + VEV_4000..VEV_6500 only (no HYDROGEL). Position limits from
  round3work/round3description.txt (extract 200, each VEV 300). Historical TTE row used in
  offline analysis is unchanged from prior notes (8/7/6d vs day index per spec; pricing not used here).
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

# Sonic: threshold in tape tick / price units (STRATEGY.txt, ORIGINAL_DISCORD_QUOTES.txt)
SPREAD_TH = 2

# When tight: two-sided passive quotes, one tick inside when spread allows
TIGHT_CLIP_VEV = 4
TIGHT_CLIP_U = 8
# When wide: no new risk; only trim inventory
WIDE_UNWIND = 2


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
        td["s_prod"] = int(max(0, s5200) * max(0, s5300))  # spread--spread co-tightness scalar
        td["tight2"] = int(joint_tight)
        td["th"] = SPREAD_TH

        orders: dict = {}

        if joint_tight:
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
                clip = TIGHT_CLIP_U if p == U else TIGHT_CLIP_VEV
                if d_u > 0.0 and p == U:
                    clip = min(lim, clip + 2)
                if d_u < 0.0 and p == U:
                    clip = max(1, clip - 1)
                buy_px, sell_px = _passive_prices(bb, ask)
                can_buy = max(0, min(clip, lim - cur))
                can_s = max(0, min(clip, lim + cur))
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
