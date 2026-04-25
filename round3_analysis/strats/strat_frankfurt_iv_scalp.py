"""Strategy G: Direct port of Frankfurt Hedgehogs IV-scalping logic to Round 3
VEV_5100 / VEV_5200 / VEV_5300.

Key Frankfurt mechanics (replicated exactly):
  - For each option compute theo from a quadratic IV smile in m_t = log(K/S)/sqrt(T).
  - theo_diff = wall_mid - theo
  - mean_theo_diff = EMA(theo_diff, window=THEO_NORM_WINDOW=20)
  - switch_mean   = EMA(|theo_diff - mean_theo_diff|, window=IV_SCALPING_WINDOW=100)
  - Trade signal (only when switch_mean >= IV_SCALPING_THR=0.7 (lowered for 5300)):
      if (current_theo_diff - wall_mid + best_bid - mean_theo_diff) >= THR_OPEN
          ASK at best_bid for max sell vol
      if (current_theo_diff - wall_mid + best_ask - mean_theo_diff) <= -THR_OPEN
          BID at best_ask for max buy vol
      Plus close-side legs when initial_position is wrong-way.

  Equivalent to: when the wall_mid - theo deviates from its EMA by enough, take
  the SAME side (lift offers / hit bids inside the wall) for max size and unwind
  on the way back.

Smile coefficients for round 3 are taken from voucher_work/overall_work/fitted_smile_coeffs.json
(already validated in round3_analysis/01_smile_analysis.py).

Critical fix vs polished.py: polished references self.new_switch_mean which is never
set; we use indicators['switch_means'] (matching the comment in 5200_work/frankfurt_iv_scalp_core.py).
"""
from __future__ import annotations

import json
import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState


UND = "VELVETFRUIT_EXTRACT"
TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIMITS = {UND: 200, "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300}

# Frankfurt knobs
THR_OPEN = 0.5
THR_CLOSE = 0.0
LOW_VEGA_THR_ADJ = 0.5
THEO_NORM_WINDOW = 20
IV_SCALPING_WINDOW = 100
# Lowered: round-3 5200 switch_mean p95 ~ 0.39 (see threshold_suggestions.txt)
IV_SCALPING_THR = 0.30

# Round 3 smile — NEAR-money (5000-5500) version, fit RMSE 0.008
SMILE = (0.029682579827555476, 0.0024113521900090236, 0.23943767718887515)
PROD_DTE_AT_OPEN = 5
STEPS_PER_DAY = 10_000
ANNUAL_TICKS = 365 * STEPS_PER_DAY


def _ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _npdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def t_years(ts: int) -> float:
    return max(1, PROD_DTE_AT_OPEN * STEPS_PER_DAY - ts // 100) / float(ANNUAL_TICKS)


def smile_iv(S, K, T):
    if S <= 0 or K <= 0 or T <= 0:
        return 0.24
    m = math.log(K / S) / math.sqrt(T)
    iv = SMILE[0] * m * m + SMILE[1] * m + SMILE[2]
    return max(iv, 0.05)


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0), 1.0 if S > K else 0.0, 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    Nd1 = _ncdf(d1); Nd2 = _ncdf(d2)
    price = S * Nd1 - K * Nd2
    delta = Nd1
    vega = S * _npdf(d1) * sqrtT
    return price, delta, vega


def book(depth):
    if depth is None:
        return {}, {}, None, None, None, None, None
    buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
    sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
    if not buys or not sells:
        return buys, sells, None, None, None, None, None
    bb = max(buys); ba = min(sells)
    bid_wall = min(buys); ask_wall = max(sells)
    wall_mid = 0.5 * (bid_wall + ask_wall)
    return buys, sells, bb, ba, bid_wall, ask_wall, wall_mid


def ema(store: dict, key: str, window: int, value: float) -> float:
    old = store.get(key, value)
    alpha = 2.0 / (window + 1.0)
    new = alpha * value + (1.0 - alpha) * old
    store[key] = new
    return new


class Trader:

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}

        try:
            store = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            store = {}

        depths = state.order_depths or {}
        positions = state.position or {}

        # Underlying mid for theo
        u_buys, u_sells, u_bb, u_ba, u_bw, u_aw, u_wm = book(depths.get(UND))
        if u_wm is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        # Frankfurt uses (best_bid + best_ask)/2 for the underlying spot
        S = 0.5 * (u_bb + u_ba)
        T = t_years(int(state.timestamp))

        ts100 = int(state.timestamp) // 100
        warmup = max(THEO_NORM_WINDOW, IV_SCALPING_WINDOW)
        in_warmup = ts100 < warmup

        for sym in TARGETS:
            depth = depths.get(sym)
            if depth is None:
                continue
            buys, sells, bb, ba, bid_wall, ask_wall, wall_mid = book(depth)

            # Frankfurt fallback: synthesize wall if missing
            if wall_mid is None:
                if ask_wall is not None and bid_wall is None:
                    wall_mid = ask_wall - 0.5
                    bid_wall = ask_wall - 1
                    bb = ask_wall - 1
                elif bid_wall is not None and ask_wall is None:
                    wall_mid = bid_wall + 0.5
                    ask_wall = bid_wall + 1
                    ba = bid_wall + 1
                else:
                    continue

            K = int(sym.split("_")[1])
            sig = smile_iv(S, K, T)
            theo, delta, vega = bs_call(S, K, T, sig)
            theo_diff = wall_mid - theo

            mean_diff = ema(store, f"{sym}_mean_diff", THEO_NORM_WINDOW, theo_diff)
            switch_mean = ema(store, f"{sym}_switch", IV_SCALPING_WINDOW,
                              abs(theo_diff - mean_diff))

            if in_warmup or bb is None or ba is None:
                continue

            pos = int(positions.get(sym, 0))
            lim = LIMITS[sym]
            max_buy = lim - pos
            max_sell = lim + pos

            ords: list[Order] = []
            low_vega_adj = LOW_VEGA_THR_ADJ if vega <= 1.0 else 0.0

            if switch_mean >= IV_SCALPING_THR:
                # SELL signal: signal = (theo_diff - wall_mid + best_bid) - mean_diff
                signal_sell = (theo_diff - wall_mid + bb) - mean_diff
                signal_buy = (theo_diff - wall_mid + ba) - mean_diff

                if signal_sell >= (THR_OPEN + low_vega_adj) and max_sell > 0:
                    ords.append(Order(sym, int(bb), -max_sell))
                    max_sell = 0
                if signal_sell >= THR_CLOSE and pos > 0:
                    qty = min(pos, lim + pos)
                    if qty > 0 and max_sell > 0:
                        ords.append(Order(sym, int(bb), -min(qty, max_sell)))
                        max_sell -= min(qty, max_sell)

                if signal_buy <= -(THR_OPEN + low_vega_adj) and max_buy > 0:
                    ords.append(Order(sym, int(ba), max_buy))
                    max_buy = 0
                if signal_buy <= -THR_CLOSE and pos < 0 and max_buy > 0:
                    qty = min(-pos, lim - pos)
                    if qty > 0:
                        ords.append(Order(sym, int(ba), min(qty, max_buy)))
                        max_buy -= min(qty, max_buy)
            else:
                # Below scalping threshold -> just unwind any inventory
                if pos > 0 and max_sell > 0:
                    ords.append(Order(sym, int(bb), -min(pos, max_sell)))
                elif pos < 0 and max_buy > 0:
                    ords.append(Order(sym, int(ba), min(-pos, max_buy)))

            if ords:
                result[sym] = ords

        return result, 0, json.dumps(store, separators=(",", ":"))
