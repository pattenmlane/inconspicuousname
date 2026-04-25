"""Strategy E: Persistently short VEV_5300 (and 5200) — they trade above smile-theo nearly all day every day. Delta-hedge."""
from __future__ import annotations

import math, json
try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UND = "VELVETFRUIT_EXTRACT"
TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIM = {"VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300, UND: 200}

# Production DTE (round 3 day 2 in our 3-day backtest is "tomorrow")
PROD_DTE = 5
ANN = 365 * 10000
SMILE_A, SMILE_B, SMILE_C = 0.14215151147708086, -0.0016298611395181932, 0.23576325646627055


def _ncdf(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def t_years(ts): return max(1, PROD_DTE*10000 - ts//100) / float(ANN)
def smile_iv(S,K,T):
    if S<=0 or K<=0 or T<=0: return 0.24
    m = math.log(K/S)/math.sqrt(T)
    return max(SMILE_A*m*m + SMILE_B*m + SMILE_C, 0.05)
def bs_delta(S,K,T,sig):
    if T<=0 or sig<=0: return 1.0 if S>K else 0.0
    d1 = (math.log(S/K)+0.5*sig*sig*T)/(sig*math.sqrt(T))
    return _ncdf(d1)


class Trader:
    def run(self, state: TradingState):
        result = {}
        positions = state.position or {}
        depths = state.order_depths or {}
        u_depth = depths.get(UND)
        if u_depth is None: return {}, 0, ""
        u_buys = {int(p): abs(int(q)) for p,q in (u_depth.buy_orders or {}).items() if int(q)!=0}
        u_sells = {int(p): abs(int(q)) for p,q in (u_depth.sell_orders or {}).items() if int(q)!=0}
        if not u_buys or not u_sells: return {}, 0, ""
        u_bb = max(u_buys); u_ba = min(u_sells); S = 0.5*(u_bb+u_ba)
        T = t_years(int(state.timestamp))
        net_delta = 0.0

        # Short each target up to a limit
        for sym in TARGETS:
            depth = depths.get(sym)
            if depth is None: continue
            buys = {int(p): abs(int(q)) for p,q in (depth.buy_orders or {}).items() if int(q)!=0}
            sells = {int(p): abs(int(q)) for p,q in (depth.sell_orders or {}).items() if int(q)!=0}
            if not buys: continue
            K = int(sym.split("_")[1])
            sig = smile_iv(S, K, T)
            d = bs_delta(S, K, T, sig)
            pos = int(positions.get(sym, 0))
            net_delta += pos * d
            # target: short -300
            target = -LIM[sym]
            need = target - pos  # negative -> need to sell more
            if need < 0:
                bb = max(buys)
                qty = min(buys[bb], -need)
                if qty > 0:
                    result.setdefault(sym, []).append(Order(sym, bb, -qty))
                    net_delta -= qty * d  # we'll be more short
        # Hedge: buy underlying long to offset short calls (calls have positive delta; short calls = negative delta; hedge = LONG underlying)
        u_target = -int(round(net_delta))
        u_target = max(-LIM[UND], min(LIM[UND], u_target))
        u_pos = int(positions.get(UND, 0))
        u_need = u_target - u_pos
        if u_need > 0:
            for sp in sorted(u_sells.keys()):
                if u_need <= 0: break
                q = min(u_sells[sp], u_need)
                result.setdefault(UND, []).append(Order(UND, sp, q))
                u_need -= q
        elif u_need < 0:
            for bp in sorted(u_buys.keys(), reverse=True):
                if u_need >= 0: break
                q = min(u_buys[bp], -u_need)
                result.setdefault(UND, []).append(Order(UND, bp, -q))
                u_need += q

        return result, 0, ""
