"""
family13 / realized_vol_regime: parent is best in-line family-13 (v10).

v15 data-led tweak: tape shows on large |Δextract| (>=3 ticks) average next-tick
spread change on 5200 is modestly more positive than calm periods (shock
microstructure; see analysis_shock_spread_v14.json). We add a *shock boost* to
the spread-stress term when the one-tick move in extract is large, making stress
hysteresis more likely to flip to the core-only regime right after microstructure
shocks without changing the RV-IV reverse-sign family logic.
"""
from __future__ import annotations

import json
import math
from collections import deque
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

U = "VELVETFRUIT_EXTRACT"
CORE = ["VEV_5000", "VEV_5100", "VEV_5200"]
BROAD = ["VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
STRIKE = {f"VEV_{k}": k for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]}

RV_WIN = 80
RV_ANNUAL = math.sqrt(365.0 * 24.0 * 60.0 * 60.0 / 100.0)
STRESS_ENTER = 0.60
STRESS_EXIT = 0.47
GAP_ENTER = 0.06
GAP_EXIT = 0.02
W_GAP = 0.9
W_STRESS = 0.1

# |ΔU| tick threshold from tape: shock episodes ~ few percent of steps
SHOCK_DU = 2.5
SHOCK_STRESS = 0.10

CALM_CLIP = 3
STRESS_CLIP = 1
CALM_MAX_POS = 80
STRESS_MAX_POS = 40


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
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if _bs_call(S, K, T, m) > price:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


class Trader:
    def run(self, state: TradingState):
        td = _parse_td(getattr(state, "traderData", None))
        mids_hist: list = td.get("u_mid_hist", [])
        if not isinstance(mids_hist, list):
            mids_hist = []

        syms = {p: _sym(state, p) for p in set(BROAD + [U])}
        if syms.get(U) is None:
            td["u_mid_hist"] = mids_hist[-RV_WIN:]
            return {}, 0, json.dumps(td, separators=(",", ":"))

        depths = getattr(state, "order_depths", None) or {}
        u_ba = _best_bid_ask(depths.get(syms[U]))
        if u_ba is None:
            td["u_mid_hist"] = mids_hist[-RV_WIN:]
            return {}, 0, json.dumps(td, separators=(",", ":"))
        u_mid = 0.5 * (u_ba[0] + u_ba[1])

        du = 0.0
        if len(mids_hist) >= 1 and mids_hist[-1] > 0:
            du = abs(u_mid - float(mids_hist[-1]))

        mids_hist.append(float(u_mid))
        mids_hist = mids_hist[-RV_WIN:]
        td["u_mid_hist"] = mids_hist

        rv = 0.0
        if len(mids_hist) >= 20:
            rets = [math.log(mids_hist[i] / mids_hist[i - 1]) for i in range(1, len(mids_hist)) if mids_hist[i - 1] > 0 and mids_hist[i] > 0]
            if len(rets) >= 10:
                mu = sum(rets) / len(rets)
                var = sum((x - mu) ** 2 for x in rets) / len(rets)
                rv = math.sqrt(max(var, 0.0)) * RV_ANNUAL

        core_mids: dict = {}
        spread_sum, spread_cnt = 0.0, 0
        for p in CORE:
            s = syms.get(p)
            if s is None:
                continue
            ba = _best_bid_ask(depths.get(s))
            if ba is None:
                continue
            core_mids[p] = 0.5 * (ba[0] + ba[1])
            spread_sum += float(ba[1] - ba[0])
            spread_cnt += 1

        if len(core_mids) < 3:
            return {}, 0, json.dumps(td, separators=(",", ":"))

        T_proxy = 7.0 / 365.0
        ivs = []
        for p in CORE:
            v = _implied_vol(float(u_mid), float(STRIKE[p]), T_proxy, float(core_mids[p]))
            if v is not None:
                ivs.append(v)
        iv_core = sum(ivs) / len(ivs) if ivs else 0.0

        avg_spread = spread_sum / max(spread_cnt, 1)
        stress_base = (avg_spread - 3.0) / 10.0
        shock = du >= SHOCK_DU
        stress_raw = stress_base + (SHOCK_STRESS if shock else 0.0)
        stress = max(0.0, min(1.5, stress_raw))

        rv_iv_gap = rv - iv_core
        signal = -(W_GAP * rv_iv_gap + W_STRESS * stress)

        regime = str(td.get("regime", "calm"))
        if regime == "calm" and stress > STRESS_ENTER:
            regime = "stress"
        elif regime == "stress" and stress < STRESS_EXIT:
            regime = "calm"
        td["regime"] = regime
        td["signal"] = round(signal, 6)
        td["rv"] = round(rv, 6)
        td["iv_core"] = round(iv_core, 6)
        td["stress"] = round(stress, 6)
        td["abs_du"] = round(du, 4)
        td["shock"] = int(shock)

        universe = CORE if regime == "stress" else BROAD
        clip = STRESS_CLIP if regime == "stress" else CALM_CLIP
        max_pos = STRESS_MAX_POS if regime == "stress" else CALM_MAX_POS

        pos = getattr(state, "position", None) or {}
        orders: dict = {}

        action = 0
        prev_action = int(td.get("action", 0))
        if prev_action == 0 and signal > GAP_ENTER:
            action = 1
        elif prev_action == 1 and signal < GAP_EXIT:
            action = 0
        elif prev_action == 0 and signal < -GAP_ENTER:
            action = -1
        elif prev_action == -1 and signal > -GAP_EXIT:
            action = 0
        else:
            action = prev_action
        td["action"] = action

        for p in universe:
            s = syms.get(p)
            if s is None:
                continue
            ba = _best_bid_ask(depths.get(s))
            if ba is None:
                continue
            bb, ask = ba
            cur = int(pos.get(s, 0))
            lim = 300
            rail = min(max_pos, lim)
            can_buy = max(0, min(clip, lim - cur))
            can_sell = max(0, min(clip, lim + cur))
            if action > 0 and can_buy > 0:
                px = bb + 1 if ask > bb + 1 else ask
                orders.setdefault(s, []).append(Order(s, int(px), int(can_buy)))
            elif action < 0 and can_sell > 0:
                px = ask - 1 if ask > bb + 1 else bb
                orders.setdefault(s, []).append(Order(s, int(px), -int(can_sell)))
            elif action == 0 and abs(cur) > rail // 3:
                if cur > 0 and can_sell > 0:
                    px = ask - 1 if ask > bb + 1 else bb
                    q = min(can_sell, max(1, abs(cur) // 4))
                    orders.setdefault(s, []).append(Order(s, int(px), -int(q)))
                elif cur < 0 and can_buy > 0:
                    px = bb + 1 if ask > bb + 1 else ask
                    q = min(can_buy, max(1, abs(cur) // 4))
                    orders.setdefault(s, []).append(Order(s, int(px), int(q)))

        return orders, 0, json.dumps(td, separators=(",", ":"))
