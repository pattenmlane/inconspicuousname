"""
Round 3 — **vouchers_final_strategy only** (no legacy smile/IV/spline stack).

- **Sonic gate:** s5200 = ask1−bid1 on VEV_5200, s5300 on VEV_5300. **Tight** = both <= 2
  (same tick units as tape / STRATEGY).
- **Regimes:** **Tight** — aggressive **VELVETFRUIT_EXTRACT** MM (small width, high size) + VEV
  two-sided around **mid** with width from **BBO spread**; extra clip on **VEV_5200** and **VEV_5300**
  (the hedging surface the quote references). **Wide** — **no extract** (avoids paying spread
  when 6-panel story does not support short-horizon mid edge) + VEV on **core strikes 5000–5500
  only** with wider, smaller quotes; **skip** deep illiquid wings in wide (spread-as-signal, inclineGod).
- **No HYDROGEL_PACK.**

DTE: not used (mid-only VEV; extract has no model fair beyond EWMA touch mid).
"""
from __future__ import annotations

import json
import math
from typing import Any

from datamodel import Order, OrderDepth, TradingState

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHERS = [f"VEV_{k}" for k in STRIKES]
U = "VELVETFRUIT_EXTRACT"
H = "HYDROGEL_PACK"

GATE_A = "VEV_5200"
GATE_B = "VEV_5300"
GATE_MAX_SPREAD = 2

# Gate legs get extra attention when tight (Sonic / tight-surface idea)
GATE_VEV_EXTRA = 6

# Tight: extract
U_MM_WIDTH_TIGHT = 1
U_QUOTE_SIZE_TIGHT = 55
U_FAIR_EWMA = 0.12

# Wide: no extract orders
# VEV: tight regime
VEV_SIZE_TIGHT = 14
VEV_SIZE_WIDE = 5
# shift = base + k * bbo_spread; then clamp to book
TIGHT_BBO_K = 0.22
WIDE_BBO_K = 0.38
TIGHT_SHIFT_BASE = 0.8
WIDE_SHIFT_BASE = 1.6

# Wide: only quote 5000–5500; deep wings off (Sonic: noise dominates)

LIMITS: dict[str, int] = {U: 200, **{v: 300 for v in VOUCHERS}}


def best_bid_ask(depth: OrderDepth) -> tuple[int | None, int | None]:
    if not depth.buy_orders or not depth.sell_orders:
        return None, None
    return max(depth.buy_orders.keys()), min(depth.sell_orders.keys())


def micro_mid(depth: OrderDepth) -> float | None:
    bb, ba = best_bid_ask(depth)
    if bb is None or ba is None:
        return None
    return 0.5 * (bb + ba)


def top_of_book_spread(dv: OrderDepth) -> int | None:
    bb, ba = best_bid_ask(dv)
    if bb is None or ba is None:
        return None
    return int(ba - bb)


def infer_csv_day(s_mid: float, h_mid: float | None) -> int:
    if abs(s_mid - 5250.0) < 1.5 and h_mid is not None and abs(h_mid - 10000.0) < 2.5:
        return 0
    if abs(s_mid - 5245.0) < 1.5 and h_mid is not None and abs(h_mid - 9958.0) < 2.5:
        return 1
    if abs(s_mid - 5267.5) < 2.0:
        return 2
    return 0


class Trader:
    def run(self, state: TradingState):
        store: dict[str, Any] = {}
        raw = getattr(state, "traderData", "") or ""
        if raw:
            try:
                o = json.loads(raw)
                if isinstance(o, dict):
                    store = o
            except (json.JSONDecodeError, TypeError):
                store = {}

        _ts = int(getattr(state, "timestamp", 0))
        pos = getattr(state, "position", {}) or {}
        depths = getattr(state, "order_depths", {}) or {}
        out: dict[str, list[Order]] = {}

        du = depths.get(U)
        if du is None:
            return out, 0, json.dumps(store, separators=(",", ":"))

        s_mid = micro_mid(du)
        if s_mid is None:
            return out, 0, json.dumps(store, separators=(",", ":"))

        d_gate_a = depths.get(GATE_A)
        d_gate_b = depths.get(GATE_B)
        sa = top_of_book_spread(d_gate_a) if d_gate_a is not None else None
        sb = top_of_book_spread(d_gate_b) if d_gate_b is not None else None
        tight = sa is not None and sb is not None and sa <= GATE_MAX_SPREAD and sb <= GATE_MAX_SPREAD

        dh = depths.get(H)
        h_mid = micro_mid(dh) if dh is not None else None
        csv_day = int(store.get("csv_day", -1))
        if csv_day < 0 or _ts == 0:
            csv_day = infer_csv_day(float(s_mid), float(h_mid) if h_mid is not None else None)
            store["csv_day"] = csv_day
        _ = csv_day  # for future; mid-only VEV

        for v in VOUCHERS:
            dv = depths.get(v)
            if dv is None:
                continue
            mid = micro_mid(dv)
            if mid is None:
                continue
            K = int(v.split("_")[1])
            if not tight and (K < 5000 or K > 5500):
                continue
            bb, ba = best_bid_ask(dv)
            if bb is None or ba is None:
                continue
            bbo = float(ba - bb)

            if tight:
                qsz = VEV_SIZE_TIGHT
                shift = TIGHT_SHIFT_BASE + TIGHT_BBO_K * bbo
                if v == GATE_A or v == GATE_B:
                    qsz = min(300, qsz + GATE_VEV_EXTRA)
            else:
                qsz = VEV_SIZE_WIDE
                shift = WIDE_SHIFT_BASE + WIDE_BBO_K * bbo

            fair = float(mid)
            pos_v = int(pos.get(v, 0))
            lim = LIMITS[v]
            bid_px = int(round(fair - shift))
            ask_px = int(round(fair + shift))
            bid_px = min(bid_px, int(ba) - 1)
            ask_px = max(ask_px, int(bb) + 1)
            if ask_px <= bid_px:
                ask_px = bid_px + 1

            q_buy = min(qsz, lim - pos_v)
            q_sell = min(qsz, lim + pos_v)
            ol: list[Order] = []
            if q_buy > 0 and bid_px > 0:
                ol.append(Order(v, bid_px, q_buy))
            if q_sell > 0:
                ol.append(Order(v, ask_px, -q_sell))
            if ol:
                out[v] = ol

        if tight:
            uf = store.get("u_fair")
            if not isinstance(uf, (int, float)) or not math.isfinite(float(uf)):
                uf = float(s_mid)
            uf = (1.0 - U_FAIR_EWMA) * float(uf) + U_FAIR_EWMA * float(s_mid)
            store["u_fair"] = uf
            pos_u = int(pos.get(U, 0))
            bb_u, ba_u = best_bid_ask(du)
            if bb_u is not None and ba_u is not None:
                ub = int(round(uf - U_MM_WIDTH_TIGHT))
                ua = int(round(uf + U_MM_WIDTH_TIGHT))
                ub = min(ub, ba_u - 1)
                ua = max(ua, bb_u + 1)
                if ua <= ub:
                    ua = ub + 1
                qu = min(U_QUOTE_SIZE_TIGHT, 200 - pos_u)
                qus = min(U_QUOTE_SIZE_TIGHT, 200 + pos_u)
                uo: list[Order] = []
                if qu > 0 and ub > 0:
                    uo.append(Order(U, ub, qu))
                if qus > 0:
                    uo.append(Order(U, ua, -qus))
                if uo:
                    out[U] = uo

        return out, 0, json.dumps(store, separators=(",", ":"))
