"""Strategy I: per-strike RELATIVE mean reversion of (mid - theo).

The smile gives an absolute theo. The market trades at a persistent offset.
Track an EMA of (mid - theo) and trade when the CURRENT deviation diverges
from its rolling mean by enough — that's a real signal even when the smile
itself is biased.

Trade actions:
  cur_dev = mid - theo
  rolling_dev = EMA(cur_dev, window=200)
  spread of dev around rolling: rolling_abs_dev = EMA(|cur - roll|, 200)
  if cur_dev > roll + Z * rolling_abs_dev: SELL at bid
  if cur_dev < roll - Z * rolling_abs_dev: BUY at ask
"""
from __future__ import annotations

import json
import math

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UND = "VELVETFRUIT_EXTRACT"
TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIMITS = {UND: 200, "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300}

# Smile (near-money fit)
SMILE = (0.029682579827555476, 0.0024113521900090236, 0.23943767718887515)
PROD_DTE = 5
ANN = 365 * 10000
WINDOW = 200
Z = 1.5
TAKE_QTY = 200
HEDGE_TOL = 5
WARMUP = 200


def _ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
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
        return {}, {}, None, None
    buys = {int(p): abs(int(q)) for p, q in (depth.buy_orders or {}).items() if int(q) != 0}
    sells = {int(p): abs(int(q)) for p, q in (depth.sell_orders or {}).items() if int(q) != 0}
    if not buys or not sells:
        return buys, sells, None, None
    return buys, sells, max(buys), min(sells)


def ema(store, key, window, value):
    old = store.get(key, value)
    a = 2.0 / (window + 1.0)
    new = a * value + (1.0 - a) * old
    store[key] = new
    return new


class Trader:

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        depths = state.order_depths or {}
        positions = state.position or {}

        try:
            store = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            store = {}
        store["n"] = int(store.get("n", 0)) + 1
        in_warmup = store["n"] < WARMUP

        u_buys, u_sells, u_bb, u_ba = book(depths.get(UND))
        if u_bb is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        S = 0.5 * (u_bb + u_ba)
        T = t_years(int(state.timestamp))

        net_delta = 0.0

        for sym in TARGETS:
            depth = depths.get(sym)
            if depth is None:
                continue
            buys, sells, bb, ba = book(depth)
            if bb is None:
                continue
            mid = 0.5 * (bb + ba)
            K = int(sym.split("_")[1])
            sig = smile_iv(S, K, T)
            theo, delta = bs_call(S, K, T, sig)
            dev = mid - theo

            roll = ema(store, f"{sym}_roll", WINDOW, dev)
            absdev = ema(store, f"{sym}_absdev", WINDOW, abs(dev - roll))

            pos = int(positions.get(sym, 0))
            net_delta += pos * delta
            lim = LIMITS[sym]
            max_buy = lim - pos
            max_sell = lim + pos

            if in_warmup:
                continue

            ords = []

            # Sell when current dev is HIGH vs rolling: market has overshot up
            up_thr = roll + Z * max(absdev, 0.3)
            dn_thr = roll - Z * max(absdev, 0.3)

            if dev > up_thr and max_sell > 0:
                qty = min(buys[bb], max_sell, TAKE_QTY)
                if qty > 0:
                    ords.append(Order(sym, int(bb), -qty))
                    net_delta -= qty * delta

            if dev < dn_thr and max_buy > 0:
                qty = min(sells[ba], max_buy, TAKE_QTY)
                if qty > 0:
                    ords.append(Order(sym, int(ba), qty))
                    net_delta += qty * delta

            # Mild unwind toward neutral when dev returns to mean
            if abs(dev - roll) < 0.2 and pos != 0:
                if pos > 0:
                    qty = min(buys[bb], pos, max_sell)
                    if qty > 0:
                        ords.append(Order(sym, int(bb), -qty))
                        net_delta -= qty * delta
                else:
                    qty = min(sells[ba], -pos, max_buy)
                    if qty > 0:
                        ords.append(Order(sym, int(ba), qty))
                        net_delta += qty * delta

            if ords:
                result[sym] = ords

        # delta hedge
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

        return result, 0, json.dumps(store, separators=(",", ":"))
