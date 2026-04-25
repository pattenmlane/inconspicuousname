"""Strategy H: take-theo-deviation (sell rich, buy cheap), delta-hedged.

For each tick on each VEV in {5100,5200,5300}:
  - compute fair theo from quadratic smile + underlying mid
  - if bid > theo + ENTRY_THR  -> hit bid (sell N contracts)
  - if ask < theo - ENTRY_THR  -> lift ask (buy N contracts)
  - if bid > theo (any positive) and we are LONG     -> close longs
  - if ask < theo (any negative) and we are SHORT    -> close shorts
  - delta-hedge net call delta on the underlying.
"""
from __future__ import annotations

import math
from typing import Any

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UND = "VELVETFRUIT_EXTRACT"
TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIMITS = {UND: 200, "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300}

# Smile (near-money fit, RMSE 0.008)
SMILE = (0.029682579827555476, 0.0024113521900090236, 0.23943767718887515)
PROD_DTE = 5
ANN = 365 * 10000

ENTRY_THR = 0.75   # bid - theo (or theo - ask) above this => take
EXIT_THR = -0.25   # cover when adverse deviation crosses this
HEDGE_TOL = 5
TAKE_QTY = 300     # max take per side per tick (capped by book + limit)


def _ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def _npdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
def t_years(ts): return max(1, PROD_DTE * 10000 - ts // 100) / float(ANN)


def smile_iv(S, K, T):
    if S <= 0 or K <= 0 or T <= 0:
        return 0.24
    m = math.log(K / S) / math.sqrt(T)
    return max(SMILE[0] * m * m + SMILE[1] * m + SMILE[2], 0.05)


def bs_call(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0), 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * sqrtT)
    d2 = d1 - sig * sqrtT
    return S * _ncdf(d1) - K * _ncdf(d2), _ncdf(d1)


def book(depth):
    if depth is None:
        return {}, {}, None, None, None, None, None
    buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
    sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
    if not buys or not sells:
        return buys, sells, None, None, None, None, None
    bb = max(buys); ba = min(sells)
    bw = min(buys); aw = max(sells)
    wm = 0.5 * (bw + aw)
    return buys, sells, bb, ba, bw, aw, wm


class Trader:

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        depths = state.order_depths or {}
        positions = state.position or {}

        u_buys, u_sells, u_bb, u_ba, _, _, _ = book(depths.get(UND))
        if u_bb is None or u_ba is None:
            return {}, 0, ""
        S = 0.5 * (u_bb + u_ba)
        T = t_years(int(state.timestamp))

        net_delta = 0.0

        for sym in TARGETS:
            depth = depths.get(sym)
            if depth is None:
                continue
            buys, sells, bb, ba, _, _, _ = book(depth)
            if bb is None:
                continue
            K = int(sym.split("_")[1])
            sig = smile_iv(S, K, T)
            theo, delta = bs_call(S, K, T, sig)

            pos = int(positions.get(sym, 0))
            lim = LIMITS[sym]
            max_buy = lim - pos
            max_sell = lim + pos
            ords = []

            bid_dev = bb - theo  # positive => bid is rich, SELL
            ask_dev = theo - ba  # positive => ask is cheap, BUY

            # ENTRY: sell when bid is RICH
            if bid_dev > ENTRY_THR and max_sell > 0:
                qty = min(buys[bb], max_sell, TAKE_QTY)
                if qty > 0:
                    ords.append(Order(sym, int(bb), -qty))
                    net_delta -= qty * delta

            # ENTRY: buy when ask is CHEAP
            if ask_dev > ENTRY_THR and max_buy > 0:
                qty = min(sells[ba], max_buy, TAKE_QTY)
                if qty > 0:
                    ords.append(Order(sym, int(ba), qty))
                    net_delta += qty * delta

            # EXIT: when long and bid > theo + EXIT_THR (still rich), close shorts/longs
            # if we're long and ask drops to theo or below, sell out
            if pos > 0 and bid_dev > EXIT_THR and max_sell > 0:
                # ladder out at the bid
                qty = min(buys[bb], pos, max_sell)
                if qty > 0 and not any(o.symbol==sym and o.quantity<0 for o in ords):
                    ords.append(Order(sym, int(bb), -qty))
                    net_delta -= qty * delta
            if pos < 0 and ask_dev > EXIT_THR and max_buy > 0:
                qty = min(sells[ba], -pos, max_buy)
                if qty > 0 and not any(o.symbol==sym and o.quantity>0 for o in ords):
                    ords.append(Order(sym, int(ba), qty))
                    net_delta += qty * delta

            # Track current position delta
            net_delta += pos * delta

            if ords:
                result[sym] = ords

        # Delta hedge underlying
        u_target = -int(round(net_delta))
        u_target = max(-LIMITS[UND], min(LIMITS[UND], u_target))
        u_pos = int(positions.get(UND, 0))
        u_need = u_target - u_pos
        if abs(u_need) >= HEDGE_TOL and u_buys and u_sells:
            if u_need > 0:
                for sp in sorted(u_sells.keys()):
                    if u_need <= 0:
                        break
                    q = min(u_sells[sp], u_need)
                    result.setdefault(UND, []).append(Order(UND, sp, q))
                    u_need -= q
            else:
                for bp in sorted(u_buys.keys(), reverse=True):
                    if u_need >= 0:
                        break
                    q = min(u_buys[bp], -u_need)
                    result.setdefault(UND, []).append(Order(UND, bp, -q))
                    u_need += q

        return result, 0, ""
