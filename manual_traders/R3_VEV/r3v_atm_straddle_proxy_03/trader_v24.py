"""
Iteration 24 — vouchers_final_strategy: joint 5200/5300 tight spread gate + K-step extract momentum (no legacy stack).

- Gate (Sonic): s(VEV_5200) <= 2 and s(VEV_5300) <= 2 (same as analyze_vev_5200_5300_tight_gate_r3.py TH=2).
- Signal: aligned with the folder's K-step forward mid analysis — we use rolling S(t)-S(t-K)
  (K=20) on extract mid, only evaluated when the joint gate is true at t (tight surface).
- inclineGod: per-leg L1 spread also gates extract (s_ex must be tradeable) and 5200/5300
  quotes use ceil/floor to cross like other traders in the repo.
- Voucher PnL: equal clipped size on 5200 and 5300 in direction of the momentum (long both calls
  if mom > 0, short both if mom < 0) plus same-direction extract. Not delta-neutral; on-voucher
  flow per shared thesis.

TTE: historical day d -> 8-d days in round3description; engine sets _prosperity4bt_hist_day.
"""
from __future__ import annotations

import json
import math
from datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"
ALL = [EXTRACT, VEV_5200, VEV_5300]
POS_LIMITS: dict[str, int] = {EXTRACT: 200, VEV_5200: 300, VEV_5300: 300}

TIGHT_TH = 2.0
MAX_EX_SPREAD = 6.0
K_MOM = 20
MOM_TH = 0.25
CLIP_EX = 60
CLIP_VEV = 12


def _mid(depth: OrderDepth) -> tuple[float, int, int] | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bb = max(depth.buy_orders)
    ba = min(depth.sell_orders)
    return (bb + ba) / 2.0, bb, ba


def _spread(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return float(min(depth.sell_orders) - max(depth.buy_orders))


class Trader:
    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        out: dict[str, list[Order]] = {k: [] for k in ALL}

        de = state.order_depths.get(EXTRACT)
        d52 = state.order_depths.get(VEV_5200)
        d53 = state.order_depths.get(VEV_5300)
        if de is None or d52 is None or d53 is None:
            return out, 0, json.dumps(td)

        m_ex = _mid(de)
        m52 = _mid(d52)
        m53 = _mid(d53)
        if m_ex is None or m52 is None or m53 is None:
            return out, 0, json.dumps(td)

        S, _, _ = m_ex
        mid52, _, _ = m52
        mid53, _, _ = m53
        s_ex = _spread(de) or 99.0
        s52 = _spread(d52) or 99.0
        s53 = _spread(d53) or 99.0

        ring = td.get("_s_ring", [])
        if not isinstance(ring, list):
            ring = []
        ring.append(float(S))
        if len(ring) > K_MOM + 1:
            ring = ring[-(K_MOM + 1) :]
        td["_s_ring"] = ring
        if len(ring) < K_MOM + 1:
            return out, 0, json.dumps(td)
        mom = float(ring[-1] - ring[0])
        if s52 > TIGHT_TH or s53 > TIGHT_TH or s_ex > MAX_EX_SPREAD or abs(mom) < MOM_TH:
            return out, 0, json.dumps(td)

        px = state.position.get(EXTRACT, 0)
        p52 = state.position.get(VEV_5200, 0)
        p53 = state.position.get(VEV_5300, 0)

        if mom > 0.0:
            qe = min(CLIP_EX, POS_LIMITS[EXTRACT] - px)
            qv = min(CLIP_VEV, POS_LIMITS[VEV_5200] - p52, POS_LIMITS[VEV_5300] - p53)
            qe = max(0, qe)
            qv = max(0, qv)
            if qe > 0:
                out[EXTRACT].append(Order(EXTRACT, int(math.ceil(S)), qe))
            if qv > 0:
                out[VEV_5200].append(Order(VEV_5200, int(math.ceil(mid52)), qv))
                out[VEV_5300].append(Order(VEV_5300, int(math.ceil(mid53)), qv))
        else:
            qe = min(CLIP_EX, POS_LIMITS[EXTRACT] + px)
            qv = min(CLIP_VEV, POS_LIMITS[VEV_5200] + p52, POS_LIMITS[VEV_5300] + p53)
            qe = max(0, qe)
            qv = max(0, qv)
            if qe > 0:
                out[EXTRACT].append(Order(EXTRACT, int(math.floor(S)), -qe))
            if qv > 0:
                out[VEV_5200].append(Order(VEV_5200, int(math.floor(mid52)), -qv))
                out[VEV_5300].append(Order(VEV_5300, int(math.floor(mid53)), -qv))

        return out, 0, json.dumps(td)
