"""
Round 3 — spread ladder convexity fade (VEV_5000 / 5100 / 5200 butterfly).

Thesis: when the 5000–5200 vertical bundle is priced with *too much* convexity vs a
rolling baseline (high z-score of mid-price butterfly), the next move in that
butterfly tends to be down — fade by selling the fly (short body, long wings).

TTE: round3description.txt maps historical tape days to TTE 8,7,6 at slice open.
Each backtest day is a fresh process; we use TTE=7d (mid of range) for BS IV used
in the bump diagnostic only — order logic is z-score on mids.

Grid (this file): Z_HI=2.0, Z_LO=0.5, WIN=500, clip_qty=18.
"""
from __future__ import annotations

import json
import math
from collections import deque
from typing import Any

try:
    from datamodel import Listing, Order, OrderDepth, TradingState
except ImportError:
    from prosperity4bt.datamodel import Listing, Order, OrderDepth, TradingState

VEV_WING_LO = "VEV_5000"
VEV_BODY = "VEV_5100"
VEV_WING_HI = "VEV_5200"
FLY_LEGS = (VEV_WING_LO, VEV_BODY, VEV_WING_HI)

# --- tuned from offline grid on Prosperity4Data/ROUND_3 (see analysis.json) ---
Z_HI = 2.0
Z_LO = 0.5
BF_WINDOW = 500
CLIP_QTY = 18
# Mid-calendar TTE for IV bump metric (days); slice-specific 8/7/6 all close to this.
TTE_DAYS_IV = 7
R_RATE = 0.0

STRIKE = {VEV_WING_LO: 5000, VEV_BODY: 5100, VEV_WING_HI: 5200}


def _sym(state: TradingState, product: str) -> str | None:
    listings: dict[str, Listing] = getattr(state, "listings", {}) or {}
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


def _best_bid_ask(depth: OrderDepth | None) -> tuple[int, int, int, int] | None:
    if depth is None:
        return None
    buys = getattr(depth, "buy_orders", None) or {}
    sells = getattr(depth, "sell_orders", None) or {}
    if not buys or not sells:
        return None
    bb = max(buys.keys())
    ba = min(sells.keys())
    if bb >= ba:
        return None
    return bb, abs(int(buys[bb])), ba, abs(int(sells[ba]))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_call(S: float, K: float, T: float, r: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / v
    d2 = d1 - v
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _implied_vol(S: float, K: float, T: float, r: float, price: float) -> float | None:
    lo, hi = 1e-4, 3.0
    if price <= max(S - K, 0.0) + 1e-6:
        return None
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        if _bs_call(S, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


class Trader:
    BF_HIST = "bf_hist"
    EMA_BUMP = "ema_bump"

    def run(self, state: TradingState):
        store = _parse_td(getattr(state, "traderData", None))
        hist: list[float] = store.get(self.BF_HIST)
        if not isinstance(hist, list):
            hist = []
        bf_deque: deque[float] = deque((float(x) for x in hist if isinstance(x, (int, float))), maxlen=BF_WINDOW)

        syms = {p: _sym(state, p) for p in FLY_LEGS}
        if any(v is None for v in syms.values()):
            store[self.BF_HIST] = list(bf_deque)
            return {}, 0, json.dumps(store, separators=(",", ":"))

        depths: dict[str, OrderDepth | None] = getattr(state, "order_depths", None) or {}
        mids: dict[str, float] = {}
        for p, sym in syms.items():
            ba = _best_bid_ask(depths.get(sym))
            if ba is None:
                store[self.BF_HIST] = list(bf_deque)
                return {}, 0, json.dumps(store, separators=(",", ":"))
            bb, _, bask, _ = ba
            mids[p] = 0.5 * (float(bb) + float(bask))

        bf = mids[VEV_WING_LO] + mids[VEV_WING_HI] - 2.0 * mids[VEV_BODY]
        bf_deque.append(bf)
        store[self.BF_HIST] = list(bf_deque)

        z = None
        if len(bf_deque) >= BF_WINDOW:
            mu = sum(bf_deque) / BF_WINDOW
            var = sum((x - mu) ** 2 for x in bf_deque) / BF_WINDOW
            std = math.sqrt(max(var, 1e-12))
            z = (bf - mu) / std

        # IV bump: body IV above average wing IV (same diagnostic as offline tape)
        u_sym = _sym(state, "VELVETFRUIT_EXTRACT")
        T = max(TTE_DAYS_IV, 1) / 365.0
        bump_ema = float(store.get(self.EMA_BUMP, 0.0))
        if u_sym is not None and z is not None:
            ud = depths.get(u_sym)
            uba = _best_bid_ask(ud)
            if uba is not None:
                ubb, _, uask, _ = uba
                S = 0.5 * (float(ubb) + float(uask))
                ivs = []
                for p in FLY_LEGS:
                    iv = _implied_vol(S, float(STRIKE[p]), T, R_RATE, mids[p])
                    if iv is None:
                        ivs = []
                        break
                    ivs.append(iv)
                if len(ivs) == 3:
                    bump = ivs[1] - 0.5 * (ivs[0] + ivs[2])
                    bump_ema = 0.05 * bump + 0.95 * bump_ema
        store[self.EMA_BUMP] = bump_ema

        pos = getattr(state, "position", None) or {}
        lim = 300

        def cap_buy(sym: str, q: int) -> int:
            return max(0, min(q, lim - int(pos.get(sym, 0))))

        def cap_sell(sym: str, q: int) -> int:
            return max(0, min(q, lim + int(pos.get(sym, 0))))

        orders: dict[str, list[Order]] = {}

        if z is None:
            return {}, 0, json.dumps(store, separators=(",", ":"))

        prev_mode = str(store.get("fly_mode", "flat"))
        if prev_mode == "flat" and z > Z_HI:
            mode = "short"
        elif prev_mode == "short" and z < Z_LO:
            mode = "flat"
        else:
            mode = prev_mode

        store["fly_mode"] = mode
        store["last_z"] = round(float(z), 4)
        store["last_bf"] = round(float(bf), 4)

        def qty_short_fly() -> int:
            q = CLIP_QTY
            return min(
                q,
                cap_sell(syms[VEV_WING_LO], CLIP_QTY),
                cap_sell(syms[VEV_WING_HI], CLIP_QTY),
                cap_buy(syms[VEV_BODY], 2 * CLIP_QTY) // 2,
            )

        def qty_cover_fly() -> int:
            q = CLIP_QTY
            return min(
                q,
                cap_buy(syms[VEV_WING_LO], CLIP_QTY),
                cap_buy(syms[VEV_WING_HI], CLIP_QTY),
                cap_sell(syms[VEV_BODY], 2 * CLIP_QTY) // 2,
            )

        # One-shot legs on mode transitions only
        if prev_mode == "flat" and mode == "short":
            q = qty_short_fly()
            if q > 0:
                wlo = _best_bid_ask(depths.get(syms[VEV_WING_LO]))
                bod = _best_bid_ask(depths.get(syms[VEV_BODY]))
                whi = _best_bid_ask(depths.get(syms[VEV_WING_HI]))
                if wlo and bod and whi:
                    wing_bid_lo = wlo[0]
                    wing_bid_hi = whi[0]
                    body_ask = bod[2]
                    orders[syms[VEV_WING_LO]] = [Order(syms[VEV_WING_LO], wing_bid_lo, -q)]
                    orders[syms[VEV_BODY]] = [Order(syms[VEV_BODY], body_ask, 2 * q)]
                    orders[syms[VEV_WING_HI]] = [Order(syms[VEV_WING_HI], wing_bid_hi, -q)]
        elif prev_mode == "short" and mode == "flat":
            q = qty_cover_fly()
            if q > 0:
                wlo = _best_bid_ask(depths.get(syms[VEV_WING_LO]))
                bod = _best_bid_ask(depths.get(syms[VEV_BODY]))
                whi = _best_bid_ask(depths.get(syms[VEV_WING_HI]))
                if wlo and bod and whi:
                    wing_ask_lo = wlo[2]
                    wing_ask_hi = whi[2]
                    body_bid = bod[0]
                    orders[syms[VEV_WING_LO]] = [Order(syms[VEV_WING_LO], wing_ask_lo, q)]
                    orders[syms[VEV_BODY]] = [Order(syms[VEV_BODY], body_bid, -2 * q)]
                    orders[syms[VEV_WING_HI]] = [Order(syms[VEV_WING_HI], wing_ask_hi, q)]

        if not orders:
            return {}, 0, json.dumps(store, separators=(",", ":"))
        return orders, 0, json.dumps(store, separators=(",", ":"))
