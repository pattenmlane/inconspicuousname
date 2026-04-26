"""
Iteration 23 — vouchers_final_strategy thesis only (no legacy smile/straddle stack).

- Sonic / STRATEGY.txt: only act when VEV_5200 and VEV_5300 L1 spreads are both <= 2 (tight
  "surface"); otherwise flat (t-stat / edge lives in the tight book state).
- inclineGod: use each contract's spread as a signal, not just mids (see ORIGINAL_DISCORD_QUOTES.txt).
- When the joint gate is on: mean reversion on VELVETFRUIT_EXTRACT mid vs EMA; optional extract-only
  so voucher legs are not the PnL drag under worse-matching. VEV_5200/VEV_5300 stay in state but are
  not actively traded in this v23 build.

TTE: round3 historical day d maps to 8-d days in round3description; backtester injects
_prosperity4bt_hist_day.
"""
from __future__ import annotations

import json
import math
from datamodel import Order, OrderDepth, TradingState

EXTRACT = "VELVETFRUIT_EXTRACT"
VEV_5200 = "VEV_5200"
VEV_5300 = "VEV_5300"

# Must cover every product the engine may have; 5200/5300 unused for orders but can carry inventory
ALL = [EXTRACT, VEV_5200, VEV_5300]

POS_LIMITS: dict[str, int] = {EXTRACT: 200, VEV_5200: 300, VEV_5300: 300}

TIGHT_TH = 2.0
MAX_EX_SPREAD = 6.0
EXTRACT_EMA_HALFLIFE = 150
EXTRACT_FADE_EDGE = 7.0
CLIP_EX = 100


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
        if m_ex is None:
            return out, 0, json.dumps(td)
        S, _, _ = m_ex

        s_ex = _spread(de) or 99.0
        s52 = _spread(d52) or 99.0
        s53 = _spread(d53) or 99.0

        if not ((s52 <= TIGHT_TH) and (s53 <= TIGHT_TH)):
            return out, 0, json.dumps(td)
        if s_ex > MAX_EX_SPREAD:
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

        if dev < -EXTRACT_FADE_EDGE:
            qe = min(CLIP_EX, POS_LIMITS[EXTRACT] - px)
            if qe > 0:
                out[EXTRACT].append(Order(EXTRACT, int(math.ceil(S)), qe))
        elif dev > EXTRACT_FADE_EDGE:
            qe = min(CLIP_EX, POS_LIMITS[EXTRACT] + px)
            if qe > 0:
                out[EXTRACT].append(Order(EXTRACT, int(math.floor(S)), -qe))

        return out, 0, json.dumps(td)
