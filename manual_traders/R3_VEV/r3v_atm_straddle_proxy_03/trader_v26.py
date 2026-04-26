"""
Iteration 26 — vouchers_final_strategy: same as v25 but **VEV_5300 only** for option legs.

v25 backtest showed VEV_5200 PnL often negative while VEV_5300 positive under the same joint gate
and equal clips; we still require s(5200) and s(5300) <= 2 (Sonic tight surface) but only cross
5300 (inclineGod: per-contract behavior differs). Extract EMA reversion unchanged from v23/v25.

No HYDROGEL_PACK.
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
SS_SUM_TIGHT = 2.0
CLIP_SCALE_TIGHT = 1.2

EXTRACT_EMA_HALFLIFE = 150
EXTRACT_FADE_EDGE = 7.0
CLIP_EX = 100
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
        _, _, _ = m52
        mid53, _, _ = m53
        s_ex = _spread(de) or 99.0
        s52 = _spread(d52) or 99.0
        s53 = _spread(d53) or 99.0

        if s52 > TIGHT_TH or s53 > TIGHT_TH or s_ex > MAX_EX_SPREAD:
            return out, 0, json.dumps(td)

        hist: list[float] = td.get("_ex_hist", [])
        hist.append(float(S))
        if len(hist) > 800:
            hist = hist[-800:]
        td["_ex_hist"] = hist

        a = 2.0 / (EXTRACT_EMA_HALFLIFE + 1.0)
        ema = float(td.get("_ex_ema", S))
        ema = (1.0 - a) * ema + a * S
        td["_ex_ema"] = ema
        dev = S - ema

        px = state.position.get(EXTRACT, 0)
        p53 = state.position.get(VEV_5300, 0)

        vclip = int(round(CLIP_VEV * (CLIP_SCALE_TIGHT if (s52 + s53) <= SS_SUM_TIGHT else 1.0)))
        vclip = max(1, min(vclip, 30))

        if dev < -EXTRACT_FADE_EDGE:
            qe = min(CLIP_EX, POS_LIMITS[EXTRACT] - px)
            qv = min(vclip, POS_LIMITS[VEV_5300] - p53)
            qe = max(0, qe)
            qv = max(0, qv)
            if qe > 0:
                out[EXTRACT].append(Order(EXTRACT, int(math.ceil(S)), qe))
            if qv > 0:
                out[VEV_5300].append(Order(VEV_5300, int(math.ceil(mid53)), qv))
        elif dev > EXTRACT_FADE_EDGE:
            qe = min(CLIP_EX, POS_LIMITS[EXTRACT] + px)
            qv = min(vclip, POS_LIMITS[VEV_5300] + p53)
            qe = max(0, qe)
            qv = max(0, qv)
            if qe > 0:
                out[EXTRACT].append(Order(EXTRACT, int(math.floor(S)), -qe))
            if qv > 0:
                out[VEV_5300].append(Order(VEV_5300, int(math.floor(mid53)), -qv))

        return out, 0, json.dumps(td)
