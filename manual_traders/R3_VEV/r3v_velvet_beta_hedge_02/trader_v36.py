"""
vouchers_final_strategy only (round3work/vouchers_final_strategy/STRATEGY.txt).

Sonic: trade when VEV_5200 and VEV_5300 BBO spreads are both <= TH (2 ticks) — hedge into
a tight surface. inclineGod: use book state (spreads) as the gate, not a parallel price-only model.
Optional directional extract: when tight, 20-tick extract mid return proxies the analysis K=20
forward-mid story (higher mean forward move when tight) — size bias with ret_20, still bid/ask execution.

No HYDROGEL (PnL objective: VEV + VELVETFRUIT_EXTRACT). No prior smile/tilt/vega-hedge stack.
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
# Gate legs + full surface when gate is on
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
    """Joint tight gate (5200+5300 BBO) size-up MM on VEV + extract; flat when wide."""

    MID_HIST = 48
    WARMUP_STEPS = 12
    TH = 2
    VEV_SIZE = 32
    VEV_SIZE_52 = 55
    U_SIZE = 40
    RET20_THRESH = 0.55
    RET20_BID_BUMP = 1.25
    RET20_ASK_CUT = 0.75

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        hist = store.get("u_mid_hist")
        if not isinstance(hist, list):
            hist = []
        hist = [float(x) for x in hist[-self.MID_HIST :] if isinstance(x, (int, float))]

        depths: dict[str, OrderDepth] = getattr(state, "order_depths", None) or {}
        pos: dict[str, int] = getattr(state, "position", None) or {}
        ts = int(getattr(state, "timestamp", 0))
        csv_day, _local_ts = _merged_day(ts)
        csv_day = min(max(csv_day, 0), 2)

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

        ret20 = 0.0
        if len(hist) >= 22:
            ret20 = float(hist[-1] - hist[-22])

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
        if not joint:
            store["u_mid_hist"] = hist
            return {}, 0, json.dumps(store, separators=(",", ":"))

        orders: dict[str, list[Order]] = {}
        for v in VOUCHERS:
            if v not in depths:
                continue
            bb, ba, wm = _book(depths[v])
            bb2, ba2, _w = _synth(bb, ba, wm)
            if ba2 <= bb2 + 1:
                continue
            vsz = self.VEV_SIZE_52 if v in (V5200, V5300) else self.VEV_SIZE
            bid_p = bb2 + 1
            ask_p = ba2 - 1
            if bid_p >= ask_p:
                continue
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
                base = float(self.U_SIZE)
                if ret20 > self.RET20_THRESH:
                    qbu = int(round(base * self.RET20_BID_BUMP))
                    qsu = int(round(base * self.RET20_ASK_CUT))
                elif ret20 < -self.RET20_THRESH:
                    qbu = int(round(base * self.RET20_ASK_CUT))
                    qsu = int(round(base * self.RET20_BID_BUMP))
                else:
                    qbu = int(base)
                    qsu = int(base)
                qu = min(qbu, max(0, LIMITS[U] - pu))
                qus = min(qsu, max(0, LIMITS[U] + pu))
                uo: list[Order] = []
                if qu > 0:
                    uo.append(Order(U, e_b, qu))
                if qus > 0:
                    uo.append(Order(U, e_a, -qus))
                if uo:
                    orders[U] = uo

        store["u_mid_hist"] = hist
        return orders, 0, json.dumps(store, separators=(",", ":"))
