"""Combined v4: combined v3 + per-strike taker against the standing book when the
mid is far from a long EMA reference (5300 mean reversion is real)."""
from __future__ import annotations

import json
import math

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Order, OrderDepth, TradingState

UND = "VELVETFRUIT_EXTRACT"
HYDRO = "HYDROGEL_PACK"
TARGETS = ["VEV_5100", "VEV_5200", "VEV_5300"]
LIMITS = {UND: 200, HYDRO: 200, "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300}

HYDRO_MEAN = 9990.0
HYDRO_BAND = 10.0
HYDRO_SKEW = 0.04
VOUCHER_SIZE = 100
VOUCHER_SOFT_CAP = 250
VOUCHER_SKEW_PER100 = 2
TAKER_THR = {"VEV_5100": 1.5, "VEV_5200": 1.0, "VEV_5300": 0.7}
TAKER_QTY = 50
EMA_WINDOW = 200
WARMUP = 50
PROD_DTE = 5
ANN = 365 * 10000
SMILE = (0.029682579827555476, 0.0024113521900090236, 0.23943767718887515)
HEDGE_TOL = 8


def _ncdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def t_years(ts): return max(1, PROD_DTE * 10000 - ts // 100) / float(ANN)


def smile_iv(S, K, T):
    if S <= 0 or K <= 0 or T <= 0:
        return 0.24
    m = math.log(K / S) / math.sqrt(T)
    return max(SMILE[0] * m * m + SMILE[1] * m + SMILE[2], 0.05)


def bs_delta(S, K, T, sig):
    if T <= 0 or sig <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T))
    return _ncdf(d1)


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


def ema(store, key, window, value):
    old = store.get(key, value)
    a = 2.0 / (window + 1.0)
    new = a * value + (1.0 - a) * old
    store[key] = new
    return new


class Trader:

    def _trade_hydrogel(self, state, result):
        depth = state.order_depths.get(HYDRO)
        if depth is None:
            return
        buys, sells, bb, ba, bw, aw, wm = book(depth)
        if wm is None:
            return
        pos = int(state.position.get(HYDRO, 0))
        max_buy = LIMITS[HYDRO] - pos
        max_sell = LIMITS[HYDRO] + pos
        passive_fair = wm - HYDRO_SKEW * pos
        ords = []
        for sp in sorted(sells.keys()):
            if max_buy <= 0: break
            if sp <= wm - 1:
                q = min(sells[sp], max_buy)
                ords.append(Order(HYDRO, sp, q)); max_buy -= q
            elif sp <= wm and pos < 0:
                q = min(sells[sp], max_buy, -pos)
                if q > 0: ords.append(Order(HYDRO, sp, q)); max_buy -= q
        for bp in sorted(buys.keys(), reverse=True):
            if max_sell <= 0: break
            if bp >= wm + 1:
                q = min(buys[bp], max_sell)
                ords.append(Order(HYDRO, bp, -q)); max_sell -= q
            elif bp >= wm and pos > 0:
                q = min(buys[bp], max_sell, pos)
                if q > 0: ords.append(Order(HYDRO, bp, -q)); max_sell -= q
        dev = wm - HYDRO_MEAN
        if dev <= -HYDRO_BAND and max_buy > 0:
            tgt = int(math.floor(min(wm + 1, HYDRO_MEAN - HYDRO_BAND / 2)))
            ords.append(Order(HYDRO, tgt, max_buy)); max_buy = 0
        elif dev >= HYDRO_BAND and max_sell > 0:
            tgt = int(math.ceil(max(wm - 1, HYDRO_MEAN + HYDRO_BAND / 2)))
            ords.append(Order(HYDRO, tgt, -max_sell)); max_sell = 0
        if max_buy > 0:
            bid_px = int(bw) + 1 if bw is not None else int(math.floor(passive_fair - 1))
            bid_px = min(bid_px, int(math.floor(passive_fair)))
            ords.append(Order(HYDRO, bid_px, max_buy))
        if max_sell > 0:
            ask_px = int(aw) - 1 if aw is not None else int(math.ceil(passive_fair + 1))
            ask_px = max(ask_px, int(math.ceil(passive_fair)))
            ords.append(Order(HYDRO, ask_px, -max_sell))
        if ords:
            result.setdefault(HYDRO, []).extend(ords)

    def _trade_vouchers(self, state, result, store):
        depths = state.order_depths or {}
        u_buys, u_sells, u_bb, u_ba, _, _, _ = book(depths.get(UND))
        if u_bb is None:
            return
        S = 0.5 * (u_bb + u_ba)
        T = t_years(int(state.timestamp))
        positions = state.position or {}
        net_delta = 0.0

        ts100 = int(state.timestamp) // 100
        in_warmup = ts100 < WARMUP

        for sym in TARGETS:
            depth = depths.get(sym)
            if depth is None:
                continue
            buys, sells, bb, ba, bw, aw, wm = book(depth)
            if bb is None:
                continue
            mid = 0.5 * (bb + ba)
            spread = ba - bb
            K = int(sym.split("_")[1])
            sig = smile_iv(S, K, T)
            d = bs_delta(S, K, T, sig)
            pos = int(positions.get(sym, 0))
            net_delta += pos * d
            lim = LIMITS[sym]
            max_buy = lim - pos
            max_sell = lim + pos
            ords = []

            # Track rolling mid for taker
            roll = ema(store, f"{sym}_roll", EMA_WINDOW, mid)
            dev = mid - roll

            # Take stale book quotes inside the wall
            wmid = (bb + ba) / 2.0
            for sp in sorted(sells.keys()):
                if max_buy <= 0: break
                if sp <= wmid - 1:
                    q = min(sells[sp], max_buy)
                    ords.append(Order(sym, sp, q)); max_buy -= q
                    net_delta += q * d
            for bp in sorted(buys.keys(), reverse=True):
                if max_sell <= 0: break
                if bp >= wmid + 1:
                    q = min(buys[bp], max_sell)
                    ords.append(Order(sym, bp, -q)); max_sell -= q
                    net_delta -= q * d

            # Mean-reversion taker (lift bid when mid is well above EMA, lift ask when below)
            if not in_warmup:
                thr = TAKER_THR.get(sym, 1.0)
                if dev >= thr and max_sell > 0:
                    qty = min(buys[bb], TAKER_QTY, max_sell)
                    if qty > 0:
                        ords.append(Order(sym, int(bb), -qty))
                        net_delta -= qty * d
                        max_sell -= qty
                elif dev <= -thr and max_buy > 0:
                    qty = min(sells[ba], TAKER_QTY, max_buy)
                    if qty > 0:
                        ords.append(Order(sym, int(ba), qty))
                        net_delta += qty * d
                        max_buy -= qty

            # Inside-the-wall passive MM
            if spread >= 3:
                bid_px = bb + 1; ask_px = ba - 1
            elif spread == 2:
                bid_px = bb; ask_px = ba
            else:
                if ords: result.setdefault(sym, []).extend(ords)
                continue
            shift = VOUCHER_SKEW_PER100 * (pos // 100)
            bid_px = max(bb, bid_px - shift)
            ask_px = max(ask_px - shift, bb + 1)
            if bid_px >= ask_px:
                bid_px = bb; ask_px = ba
            buy_q = min(VOUCHER_SIZE, max_buy, max(0, VOUCHER_SOFT_CAP - pos))
            sell_q = min(VOUCHER_SIZE, max_sell, max(0, VOUCHER_SOFT_CAP + pos))
            if buy_q > 0:
                ords.append(Order(sym, bid_px, buy_q))
            if sell_q > 0:
                ords.append(Order(sym, ask_px, -sell_q))
            if ords:
                result.setdefault(sym, []).extend(ords)

        # Hedge
        u_target = -int(round(net_delta))
        u_target = max(-LIMITS[UND], min(LIMITS[UND], u_target))
        u_pos = int(positions.get(UND, 0))
        u_need = u_target - u_pos
        if abs(u_need) >= HEDGE_TOL and u_buys and u_sells:
            if u_need > 0:
                for sp in sorted(u_sells.keys()):
                    if u_need <= 0: break
                    q = min(u_sells[sp], u_need)
                    result.setdefault(UND, []).append(Order(UND, sp, q))
                    u_need -= q
            else:
                for bp in sorted(u_buys.keys(), reverse=True):
                    if u_need >= 0: break
                    q = min(u_buys[bp], -u_need)
                    result.setdefault(UND, []).append(Order(UND, bp, -q))
                    u_need += q

    def _mm_underlying(self, state, result):
        depth = state.order_depths.get(UND)
        if depth is None: return
        buys, sells, bb, ba, bw, aw, wm = book(depth)
        if wm is None: return
        pos = int(state.position.get(UND, 0))
        queued = sum(o.quantity for o in result.get(UND, []))
        max_buy = max(0, LIMITS[UND] - (pos + queued))
        max_sell = max(0, LIMITS[UND] + (pos + queued))
        ords = []
        for sp in sorted(sells.keys()):
            if max_buy <= 0: break
            if sp < bw:
                q = min(sells[sp], max_buy)
                ords.append(Order(UND, sp, q)); max_buy -= q
        for bp in sorted(buys.keys(), reverse=True):
            if max_sell <= 0: break
            if bp > aw:
                q = min(buys[bp], max_sell)
                ords.append(Order(UND, bp, -q)); max_sell -= q
        skew = 0.02 * pos
        passive = wm - skew
        if max_buy > 0 and bw is not None:
            bid_px = int(bw) + 1
            if bid_px < passive:
                ords.append(Order(UND, bid_px, min(max_buy, 30)))
        if max_sell > 0 and aw is not None:
            ask_px = int(aw) - 1
            if ask_px > passive:
                ords.append(Order(UND, ask_px, -min(max_sell, 30)))
        if ords:
            result.setdefault(UND, []).extend(ords)

    def run(self, state: TradingState):
        result = {}
        try:
            store = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            store = {}
        try:
            self._trade_hydrogel(state, result)
        except Exception as e:
            print(f"hydro err {e}")
        try:
            self._trade_vouchers(state, result, store)
        except Exception as e:
            print(f"vouch err {e}")
        try:
            self._mm_underlying(state, result)
        except Exception as e:
            print(f"und err {e}")
        return result, 0, json.dumps(store, separators=(",", ":"))
