"""
family13 / realized_vol_regime variation B (duplicate-risk-adjusted):
- Signal blends RV-IV gap with spread+depth stress, but with different assumptions:
  * RV proxy: EWMA abs-return volatility (robust to jumps) on extract mids.
  * IV proxy: weighted core IV (5100 gets higher weight).
  * Stress proxy: normalized spread + top-level depth-thinning metric.
- Stressed regime: core strikes only + tighter rails + only act on larger signal.
- Calm regime: broader participation + wider rails.

This is intentionally distinct from v10's direct RV std + simple spread stress blend.
"""
from __future__ import annotations

import json
import math
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
CORE = ["VEV_5000", "VEV_5100", "VEV_5200"]
BROAD = ["VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
STRIKE = {f"VEV_{k}": k for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]}

RV_WIN = 90
EWMA_ALPHA = 0.08
SCALE_ANNUAL = math.sqrt(365.0 * 24.0 * 60.0 * 60.0 / 100.0)

STRESS_ENTER = 0.58
STRESS_EXIT = 0.45
GAP_ENTER_CALM = 0.07
GAP_EXIT_CALM = 0.025
GAP_ENTER_STRESS = 0.10
GAP_EXIT_STRESS = 0.040
W_GAP = 0.85
W_STRESS = 0.15

CALM_CLIP = 2
STRESS_CLIP = 1
CALM_MAX_POS = 55
STRESS_MAX_POS = 30


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


def _best_book(depth: OrderDepth | None):
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
    bidv = abs(int(buys[bb]))
    askv = abs(int(sells[ba]))
    return int(bb), int(ba), bidv, askv


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def _implied_vol(S: float, K: float, T: float, price: float) -> float | None:
    if price <= max(S - K, 0.0) + 1e-6:
        return None
    lo, hi = 1e-4, 3.0
    for _ in range(48):
        m = 0.5 * (lo + hi)
        if _bs_call(S, K, T, m) > price:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        u_hist = td.get("u_mid_hist", [])
        if not isinstance(u_hist, list):
            u_hist = []

        syms = {p: _sym(state, p) for p in set(BROAD + [U])}
        if syms.get(U) is None:
            td["u_mid_hist"] = u_hist[-RV_WIN:]
            return {}, 0, json.dumps(td, separators=(",", ":"))

        depths = getattr(state, "order_depths", None) or {}
        ub = _best_book(depths.get(syms[U]))
        if ub is None:
            td["u_mid_hist"] = u_hist[-RV_WIN:]
            return {}, 0, json.dumps(td, separators=(",", ":"))
        u_mid = 0.5 * (ub[0] + ub[1])
        u_hist.append(float(u_mid))
        u_hist = u_hist[-RV_WIN:]
        td["u_mid_hist"] = u_hist

        rv = 0.0
        if len(u_hist) >= 20:
            rets = [abs(math.log(u_hist[i] / u_hist[i - 1])) for i in range(1, len(u_hist)) if u_hist[i - 1] > 0 and u_hist[i] > 0]
            ew = 0.0
            for r in rets:
                ew = EWMA_ALPHA * r + (1 - EWMA_ALPHA) * ew
            rv = ew * SCALE_ANNUAL

        core_mid, spread_sum, depth_inv_sum = {}, 0.0, 0.0
        cnt = 0
        for p in CORE:
            s = syms.get(p)
            if s is None:
                continue
            b = _best_book(depths.get(s))
            if b is None:
                continue
            bb, ba, bv, av = b
            core_mid[p] = 0.5 * (bb + ba)
            spread_sum += float(ba - bb)
            depth_inv_sum += 1.0 / max(1.0, float(bv + av))
            cnt += 1
        if len(core_mid) < 3:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        T = 7.0 / 365.0
        iv = {}
        for p in CORE:
            v = _implied_vol(float(u_mid), float(STRIKE[p]), T, float(core_mid[p]))
            if v is None:
                return {}, 0, json.dumps(td, separators=(",", ":"))
            iv[p] = v

        iv_core = 0.25 * iv["VEV_5000"] + 0.50 * iv["VEV_5100"] + 0.25 * iv["VEV_5200"]
        avg_spread = spread_sum / max(1, cnt)
        depth_thin = depth_inv_sum / max(1, cnt)
        stress = max(0.0, min(1.5, 0.08 * (avg_spread - 2.5) + 40.0 * depth_thin))

        signal = -(W_GAP * (rv - iv_core) + W_STRESS * stress)

        regime = str(td.get("regime", "calm"))
        if regime == "calm" and stress > STRESS_ENTER:
            regime = "stress"
        elif regime == "stress" and stress < STRESS_EXIT:
            regime = "calm"
        td["regime"] = regime

        td["rv"] = round(rv, 6)
        td["iv_core"] = round(iv_core, 6)
        td["stress"] = round(stress, 6)
        td["signal"] = round(signal, 6)

        universe = CORE if regime == "stress" else BROAD
        clip = STRESS_CLIP if regime == "stress" else CALM_CLIP
        max_pos = STRESS_MAX_POS if regime == "stress" else CALM_MAX_POS
        enter = GAP_ENTER_STRESS if regime == "stress" else GAP_ENTER_CALM
        exit_ = GAP_EXIT_STRESS if regime == "stress" else GAP_EXIT_CALM

        action_prev = int(td.get("action", 0))
        if action_prev == 0 and signal > enter:
            action = 1
        elif action_prev == 0 and signal < -enter:
            action = -1
        elif action_prev == 1 and signal < exit_:
            action = 0
        elif action_prev == -1 and signal > -exit_:
            action = 0
        else:
            action = action_prev
        td["action"] = action

        pos = getattr(state, "position", None) or {}
        orders = {}
        for p in universe:
            s = syms.get(p)
            if s is None:
                continue
            b = _best_book(depths.get(s))
            if b is None:
                continue
            bb, ba, _, _ = b
            cur = int(pos.get(s, 0))
            rail = min(300, max_pos)
            can_buy = max(0, min(clip, rail - cur))
            can_sell = max(0, min(clip, rail + cur))
            if action > 0 and can_buy > 0:
                px = bb + 1 if ba > bb + 1 else ba
                orders.setdefault(s, []).append(Order(s, int(px), int(can_buy)))
            elif action < 0 and can_sell > 0:
                px = ba - 1 if ba > bb + 1 else bb
                orders.setdefault(s, []).append(Order(s, int(px), -int(can_sell)))
            elif action == 0 and abs(cur) > rail // 3:
                if cur > 0 and can_sell > 0:
                    q = min(can_sell, max(1, abs(cur)//3))
                    px = ba - 1 if ba > bb + 1 else bb
                    orders.setdefault(s, []).append(Order(s, int(px), -int(q)))
                elif cur < 0 and can_buy > 0:
                    q = min(can_buy, max(1, abs(cur)//3))
                    px = bb + 1 if ba > bb + 1 else ba
                    orders.setdefault(s, []).append(Order(s, int(px), int(q)))

        return orders, 0, json.dumps(td, separators=(",", ":"))
