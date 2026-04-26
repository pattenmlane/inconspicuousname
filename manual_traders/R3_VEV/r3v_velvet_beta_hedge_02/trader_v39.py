"""
vouchers_final_strategy — v39 (parent v38).

v38 sizing + inclineGod-style spread co-tightness: when joint gate holds, if the *sum* of
the two gate-leg BBO spreads is small (both in the scatter's low-left), extra size — models
correlation of spreads, not mid alone.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
V5200 = "VEV_5200"
V5300 = "VEV_5300"
STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
LIMITS = {U: 200, **{v: 300 for v in VOUCHERS}}


def _book(depth: OrderDepth) -> tuple[int | None, int | None, float | None]:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys and not sells:
        return None, None, None
    bb = max(buys.keys())
    ba = min(sells.keys())
    bw = min(buys.keys())
    aw = max(sells.keys())
    return bb, ba, (float(bw) + float(aw)) / 2.0


def _bbo_spread(depth: OrderDepth) -> int | None:
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if ba <= bb:
        return None
    return int(ba - bb)


def _synth(bb: int | None, ba: int | None, wm: float | None) -> tuple[int, int, float]:
    if wm is not None and bb is not None and ba is not None:
        return bb, ba, wm
    if ba is not None and bb is None:
        return int(ba) - 1, int(ba), float(ba) - 0.5
    if bb is not None and ba is None:
        return int(bb), int(bb) + 1, float(bb) + 0.5
    return 0, 0, 0.0


def _parse_td(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        o = json.loads(raw)
        return o if isinstance(o, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _merged_day(ts: int) -> tuple[int, int]:
    return int(ts) // 1_000_000, int(ts) % 1_000_000


class Trader:
    MID_HIST = 48
    WARMUP_STEPS = 12
    TH = 2
    BASE_VEV = 36
    BASE_VEV_ATM = 52
    BASE_U = 48
    SIZE_MULT_TIGHT = 1.58
    SIZE_MULT_WIDE = 0.72
    # Both legs minimal spread (low-left in s5200 vs s5300 scatter) — extra boost
    CO_SPREAD_SUM_MAX = 3
    CO_SPREAD_MULT = 1.12
    RET20_THRESH = 0.45
    RET20_BID = 1.2
    RET20_ASK = 0.8

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        hist = store.get("u_mid_hist")
        if not isinstance(hist, list):
            hist = []
        hist = [float(x) for x in hist[-self.MID_HIST :] if isinstance(x, (int, float))]

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}
        ts = int(getattr(state, "timestamp", 0))
        _csv_d, _lt = _merged_day(ts)

        if U not in depths:
            store["u_mid_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        bb_u, ba_u, _ = _book(depths[U])
        if bb_u is None or ba_u is None:
            store["u_mid_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        u_mid = 0.5 * (float(bb_u) + float(ba_u))
        hist.append(u_mid)
        hist = hist[-self.MID_HIST :]
        ret20 = float(hist[-1] - hist[-22]) if len(hist) >= 22 else 0.0

        if ts // 100 < self.WARMUP_STEPS:
            store["u_mid_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        s52 = _bbo_spread(depths[V5200]) if V5200 in depths else None
        s53 = _bbo_spread(depths[V5300]) if V5300 in depths else None
        joint = (
            s52 is not None
            and s53 is not None
            and s52 <= self.TH
            and s53 <= self.TH
        )
        sm = self.SIZE_MULT_TIGHT if joint else self.SIZE_MULT_WIDE
        if joint and s52 is not None and s53 is not None and (s52 + s53) <= self.CO_SPREAD_SUM_MAX:
            sm *= self.CO_SPREAD_MULT

        orders: dict[str, list[Order]] = {}
        for v in VOUCHERS:
            if v not in depths:
                continue
            bb, ba, wm = _book(depths[v])
            bb2, ba2, _w = _synth(bb, ba, wm)
            if ba2 <= bb2 + 1:
                continue
            bid_p = bb2 + 1
            ask_p = ba2 - 1
            if bid_p >= ask_p:
                continue
            base = self.BASE_VEV_ATM if v in (V5200, V5300) else self.BASE_VEV
            vsz = max(6, int(round(base * sm)))
            p0 = int(pos.get(v, 0))
            lim = LIMITS[v]
            qb = min(vsz, max(0, lim - p0))
            qs = min(vsz, max(0, lim + p0))
            lo: list[Order] = []
            if qb > 0:
                lo.append(Order(v, bid_p, qb))
            if qs > 0:
                lo.append(Order(v, ask_p, -qs))
            if lo:
                orders[v] = lo

        bbu, bau, _ = _book(depths[U])
        if bbu is not None and bau is not None and bau > bbu + 1:
            e_b = int(bbu) + 1
            e_a = int(bau) - 1
            if e_b < e_a:
                pu = int(pos.get(U, 0))
                bu = max(6, int(round(self.BASE_U * sm)))
                if ret20 > self.RET20_THRESH:
                    qu = min(int(round(bu * self.RET20_BID)), max(0, LIMITS[U] - pu))
                    qus = min(int(round(bu * self.RET20_ASK)), max(0, LIMITS[U] + pu))
                elif ret20 < -self.RET20_THRESH:
                    qu = min(int(round(bu * self.RET20_ASK)), max(0, LIMITS[U] - pu))
                    qus = min(int(round(bu * self.RET20_BID)), max(0, LIMITS[U] + pu))
                else:
                    qu = min(bu, max(0, LIMITS[U] - pu))
                    qus = min(bu, max(0, LIMITS[U] + pu))
                uo: list[Order] = []
                if qu > 0:
                    uo.append(Order(U, e_b, qu))
                if qus > 0:
                    uo.append(Order(U, e_a, -qus))
                if uo:
                    orders[U] = uo

        store["u_mid_hist"] = hist
        return orders, 0, json.dumps(store, separators=(",", ":"))
