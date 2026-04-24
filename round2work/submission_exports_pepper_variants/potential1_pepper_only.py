"""Pepper leg only, extracted from ``round2work/litests/potential1.py`` (v27).

Same ``_pepper`` logic and ``traderData`` keys as the full potential1 trader.
"""

from __future__ import annotations

import json

from datamodel import Order, OrderDepth, TradingState

PEPPER = "INTARIAN_PEPPER_ROOT"


class Trader:
    LIMITS = {PEPPER: 80}

    PEP_DRIFT = 0.100188
    PEP_ENTRY_TAKE = 7
    PEP_ENTRY_TIMEOUT = 200
    PEP_BID_FLOOR = -6
    PEP_BID_CEIL = 5

    def bid(self) -> int:
        return 1103

    def run(self, state: TradingState):
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}
        result: dict[str, list[Order]] = {}
        depth = state.order_depths.get(PEPPER)
        if depth is not None:
            result[PEPPER] = self._pepper(
                PEPPER, depth, state.position.get(PEPPER, 0), state.timestamp, td
            )
        return result, 0, json.dumps(td)

    def _pepper(self, symbol: str, depth: OrderDepth, pos: int, timestamp: int, td: dict):
        if not depth.buy_orders or not depth.sell_orders:
            return []
        lim = self.LIMITS[PEPPER]
        tick = timestamp // 100

        bb = max(depth.buy_orders)
        ba = min(depth.sell_orders)
        bv_tob = depth.buy_orders[bb]
        av_tob = -depth.sell_orders[ba]
        tot = bv_tob + av_tob
        mid = (bb * av_tob + ba * bv_tob) / tot if tot > 0 else (bb + ba) / 2.0

        pep_sum = td.get("_pep_sum", 0.0) + mid - self.PEP_DRIFT * tick
        pep_cnt = td.get("_pep_cnt", 0) + 1
        td["_pep_sum"] = pep_sum
        td["_pep_cnt"] = pep_cnt

        fair = pep_sum / pep_cnt + self.PEP_DRIFT * tick
        fair_int = int(round(fair))

        need = lim - pos
        if need <= 0:
            return []

        orders = []
        bv = 0
        selective = tick < self.PEP_ENTRY_TIMEOUT
        threshold = fair + self.PEP_ENTRY_TAKE if selective else float("inf")

        for a in sorted(depth.sell_orders):
            if bv >= need or a > threshold:
                break
            vol = min(-depth.sell_orders[a], need - bv)
            if vol > 0:
                orders.append(Order(symbol, a, vol))
                bv += vol

        if selective and bv < need:
            competing = max(depth.buy_orders)
            offset = max(self.PEP_BID_FLOOR, min(self.PEP_BID_CEIL, competing + 1 - fair_int))
            orders.append(Order(symbol, fair_int + offset, need - bv))

        return orders
