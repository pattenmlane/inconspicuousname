"""
Round 3 — vouchers_final_strategy: joint VEV_5200 + VEV_5300 tight BBO spread gate (TH=2).

When BOTH spreads (ask1-bid1) are <= 2: risk-on — quote extract + all VEVs with smaller half-spread
and larger size (Sonic: hedge into a tight surface; inclineGod: book state, not just mid).

When either book is wide: risk-off — minimal size and wide quotes (do not trust small edge).

No HYDROGEL_PACK, no smile/IV-RV; see round3work/vouchers_final_strategy/STRATEGY.txt.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
# Tradeable only (PnL focus); hydrogel excluded.
LIMITS = {U: 200, **{v: 300 for v in VOUCHERS}}


def book_walls(depth: OrderDepth) -> tuple[int | None, int | None, int | None, int | None]:
    buys = depth.buy_orders or {}
    sells = depth.sell_orders or {}
    if not buys and not sells:
        return None, None, None, None
    bid_wall = min(buys.keys())
    sell_prices = list(sells.keys())
    ask_wall = max(sell_prices) if sell_prices else None
    best_bid = max(buys.keys())
    best_ask = min(sell_prices) if sell_prices else None
    return bid_wall, ask_wall, best_bid, best_ask


def micro_mid(depth: OrderDepth) -> float | None:
    _, _, bb, ba = book_walls(depth)
    if bb is None or ba is None:
        return None
    return (float(bb) + float(ba)) / 2.0


def bbo_spread(depth: OrderDepth) -> float | None:
    _, _, bb, ba = book_walls(depth)
    if bb is None or ba is None:
        return None
    return float(ba) - float(bb)


class Trader:
    # Gate (STRATEGY.txt / Sonic): both legs <= this many ticks
    TIGHT_SPREAD_TH = 2.0
    EMA_S = 120

    # Risk-on: tight half-spreads, large clips (capped by limits in run())
    TIGHT_VEV_HALF = 1.5
    TIGHT_EX_HALF = 1.5
    TIGHT_VEV_SIZE = 55
    TIGHT_EX_SIZE = 55

    # Risk-off: wide, small — execution dominates per Sonic
    WIDE_VEV_HALF = 6.0
    WIDE_EX_HALF = 5.0
    WIDE_VEV_SIZE = 4
    WIDE_EX_SIZE = 3

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conv = 0
        raw = getattr(state, "traderData", "") or ""
        try:
            td: dict[str, Any] = json.loads(raw) if str(raw).strip() else {}
        except (json.JSONDecodeError, TypeError):
            td = {}

        csv_day = int(getattr(state, "_prosperity4bt_csv_day", td.get("csv_day", 0)))
        td["csv_day"] = csv_day
        depths: dict = getattr(state, "order_depths", {}) or {}

        def sym(product: str) -> str | None:
            listings = getattr(state, "listings", {}) or {}
            for s, lst in listings.items():
                if getattr(lst, "product", None) == product:
                    return s
            return product if product in depths else None

        du = sym(U)
        if du is None or du not in depths:
            return result, conv, json.dumps(td, separators=(",", ":"))

        s5200 = sym("VEV_5200")
        s5300 = sym("VEV_5300")
        if s5200 is None or s5300 is None or s5200 not in depths or s5300 not in depths:
            return result, conv, json.dumps(td, separators=(",", ":"))

        sp5200 = bbo_spread(depths[s5200])
        sp5300 = bbo_spread(depths[s5300])
        if sp5200 is None or sp5300 is None:
            return result, conv, json.dumps(td, separators=(",", ":"))

        th = float(self.TIGHT_SPREAD_TH)
        joint_tight = sp5200 <= th and sp5300 <= th
        td["joint_tight_5200_5300"] = joint_tight
        td["s5200_bbo_spread"] = float(sp5200)
        td["s5300_bbo_spread"] = float(sp5300)
        # Spread–spread: store product for analysis logs
        td["spread_5200_x_5300"] = float(sp5200) * float(sp5300)

        d_u: OrderDepth = depths[du]
        mid_u = micro_mid(d_u)
        if mid_u is None:
            return result, conv, json.dumps(td, separators=(",", ":"))

        alpha = 2.0 / (float(self.EMA_S) + 1.0)
        ema = float(td.get("ema_s", mid_u))
        ema = alpha * mid_u + (1.0 - alpha) * ema
        td["ema_s"] = ema

        if joint_tight:
            half_u = float(self.TIGHT_EX_HALF)
            half_v = float(self.TIGHT_VEV_HALF)
            qv = int(self.TIGHT_VEV_SIZE)
            qu = int(self.TIGHT_EX_SIZE)
        else:
            half_u = float(self.WIDE_EX_HALF)
            half_v = float(self.WIDE_VEV_HALF)
            qv = int(self.WIDE_VEV_SIZE)
            qu = int(self.WIDE_EX_SIZE)

        pos: dict = getattr(state, "position", {}) or {}

        # VEVs: single fair = micro, regime sets width + size
        for v in VOUCHERS:
            svv = sym(v)
            if svv is None or svv not in depths:
                continue
            d = depths[svv]
            m = micro_mid(d)
            if m is None:
                continue
            bb, ba = book_walls(d)[2], book_walls(d)[3]
            if bb is None or ba is None:
                continue
            theo = float(m)
            p = int(pos.get(svv, 0))
            lim = LIMITS[v]
            bid_p = int(round(theo - half_v))
            ask_p = int(round(theo + half_v))
            bid_p = min(bid_p, int(ba) - 1)
            ask_p = max(ask_p, int(bb) + 1)
            if bid_p >= ask_p:
                continue
            q = max(1, min(qv, 300))
            if p < lim:
                result.setdefault(svv, []).append(Order(svv, bid_p, min(q, lim - p)))
            if p > -lim:
                result.setdefault(svv, []).append(Order(svv, ask_p, -min(q, lim + p)))

        # Extract: MM around EMA
        pu = int(pos.get(du, 0))
        lim_u = LIMITS[U]
        bu = int(round(ema - half_u))
        au = int(round(ema + half_u))
        bbu, bau = book_walls(d_u)[2], book_walls(d_u)[3]
        if bbu is not None and bau is not None:
            bu = min(bu, int(bau) - 1)
            au = max(au, int(bbu) + 1)
        qu = max(1, min(qu, 200))
        if bu < au:
            if pu < lim_u:
                result.setdefault(du, []).append(Order(du, bu, min(qu, lim_u - pu)))
            if pu > -lim_u:
                result.setdefault(du, []).append(Order(du, au, -min(qu, lim_u + pu)))

        return result, conv, json.dumps(td, separators=(",", ":"))
