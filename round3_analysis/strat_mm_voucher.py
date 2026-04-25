"""Minimal candidate strategy: 2-sided passive market making on VEV_5100/5200/5300.

Key design choices motivated by microstructure:
  - 5100 spread = 4 ticks (98%): post at bid+1 / ask-1 -> capture 2 ticks per round-trip.
  - 5200 spread = 3 ticks (80%): post at bid+1 / ask-1 -> capture 1 tick per round-trip.
  - 5300 spread = 2 ticks (84%): can't step inside; quote at bid / ask exactly to join queue.

Inventory skew: shift quoting fair by SKEW * pos so we lean against our position.

Optional delta hedge on VELVETFRUIT_EXTRACT.
"""
from __future__ import annotations

import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UNDERLYING = "VELVETFRUIT_EXTRACT"
TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]

POS_LIM = {
    UNDERLYING: 200,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
}

# How aggressive to push inside the wall:
#   spread 4: post at bid+1 / ask-1 (always)
#   spread 3: post at bid+1 / ask-1 (always)
#   spread 2: post at bid / ask (join the touch on both sides)
#   spread 1: stand aside (no profitable spot)

# Inventory skew (in ticks per unit) — lean fair toward 0
SKEW_PER_UNIT = 0.04
# Cap per strike (well below 300 hard limit) to limit single-strike inventory
STRIKE_SOFT_CAP = 200
# Quote size per side
QUOTE_SIZE = 30

# Delta hedge config
HEDGE_ENABLED = True
HEDGE_TOL = 10  # only sweep when |net delta| > HEDGE_TOL


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    return _norm_cdf(d1)


# Smile constants (for delta computation only)
SMILE_A = 0.14215151147708086
SMILE_B = -0.0016298611395181932
SMILE_C = 0.23576325646627055
PRODUCTION_DTE_AT_OPEN = 5
STEPS_PER_DAY = 10_000
ANNUAL_TICKS = 365 * STEPS_PER_DAY


def t_years_production(ts: int) -> float:
    step = ts // 100
    rem = PRODUCTION_DTE_AT_OPEN * STEPS_PER_DAY - step
    return max(rem, 1) / float(ANNUAL_TICKS)


def smile_iv(S, K, T):
    if S <= 0 or K <= 0 or T <= 0:
        return 0.24
    m = math.log(K / S) / math.sqrt(T)
    return max(SMILE_A * m * m + SMILE_B * m + SMILE_C, 0.05)


def book(depth: OrderDepth):
    if depth is None:
        return None, None, None, None, None, None
    buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
    sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
    if not buys or not sells:
        return buys, sells, None, None, None, None
    bb = max(buys); ba = min(sells)
    spread = ba - bb
    mid = 0.5 * (bb + ba)
    return buys, sells, bb, ba, spread, mid


class Trader:

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        depths = state.order_depths or {}
        positions = state.position or {}

        # Underlying mid for delta hedge
        u_depth = depths.get(UNDERLYING)
        u_buys, u_sells, u_bb, u_ba, u_spread, u_mid = book(u_depth) if u_depth else (None, None, None, None, None, None)
        T = t_years_production(int(state.timestamp))

        net_delta = 0.0

        for sym in TARGETS:
            depth = depths.get(sym)
            if depth is None:
                continue
            buys, sells, bb, ba, spread, mid = book(depth)
            if bb is None:
                continue

            K = int(sym.split("_")[1])
            pv = int(positions.get(sym, 0))
            lim = POS_LIM[sym]

            # Track delta of current position
            if u_mid is not None and T > 0:
                sig = smile_iv(u_mid, K, T)
                d = bs_delta(u_mid, K, T, sig)
                net_delta += pv * d
            else:
                d = 0.5

            # Inventory skew on quote prices
            skew = SKEW_PER_UNIT * pv  # positive when long => push quotes down
            ords: list[Order] = []

            # Choose quote prices based on spread
            if spread >= 3:
                bid_px = bb + 1
                ask_px = ba - 1
            elif spread == 2:
                # Step inside is impossible; join the touch
                bid_px = bb
                ask_px = ba
            else:
                # spread 1 — never trade
                continue

            # Apply skew: if long, lower both bid and ask; if short, raise both
            if skew > 0.5:
                bid_px = max(bb, bid_px - 1)
                ask_px = max(ask_px - 1, bb + 1)
            elif skew < -0.5:
                bid_px = min(ba - 1, bid_px + 1)
                ask_px = min(ba, ask_px + 1)

            # Make sure we're still inside the touch and not crossing
            if bid_px >= ask_px:
                bid_px = bb
                ask_px = ba
            if bid_px > bb + (spread - 1):
                bid_px = bb + (spread - 1)
            if ask_px < ba - (spread - 1):
                ask_px = ba - (spread - 1)

            # Position management: cap soft limit
            buy_qty = min(QUOTE_SIZE, lim - pv, max(0, STRIKE_SOFT_CAP - pv))
            sell_qty = min(QUOTE_SIZE, lim + pv, max(0, STRIKE_SOFT_CAP + pv))

            if buy_qty > 0:
                ords.append(Order(sym, int(bid_px), int(buy_qty)))
            if sell_qty > 0:
                ords.append(Order(sym, int(ask_px), -int(sell_qty)))

            if ords:
                result.setdefault(sym, []).extend(ords)

            # Also: take any clearly mispriced sweeps (price beats theo by >2).
            # But theo is uncertain; skip for now.

        # Delta hedge net option position on the underlying
        if HEDGE_ENABLED and u_mid is not None:
            target_u = -int(round(net_delta))
            target_u = max(-POS_LIM[UNDERLYING], min(POS_LIM[UNDERLYING], target_u))
            cur_u = int(positions.get(UNDERLYING, 0))
            need = target_u - cur_u
            if abs(need) >= HEDGE_TOL:
                u_ords: list[Order] = []
                if need > 0 and u_sells:
                    remaining = need
                    for sp in sorted(u_sells.keys()):
                        if remaining <= 0:
                            break
                        q = min(u_sells[sp], remaining)
                        u_ords.append(Order(UNDERLYING, sp, q))
                        remaining -= q
                elif need < 0 and u_buys:
                    remaining = -need
                    for bp in sorted(u_buys.keys(), reverse=True):
                        if remaining <= 0:
                            break
                        q = min(u_buys[bp], remaining)
                        u_ords.append(Order(UNDERLYING, bp, -q))
                        remaining -= q
                if u_ords:
                    result.setdefault(UNDERLYING, []).extend(u_ords)

        return result, 0, state.traderData or ""
